"""Training pipeline for the OTTO multi-objective recommender.

The implementation is designed to train on a CPU workstation with bounded
memory:

* leakage-safe, time-based pseudo test set;
* three target-sensitive, directed co-visitation heavy-hitter matrices;
* repeat, recency, type and global-popularity candidates;
* separate LightGBM LambdaRank models for clicks, carts and orders;
* aggregate weighted Recall@20 evaluation.

It streams the 11 GB jsonl file and never materializes all events in pandas.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numba as nb
import numpy as np
import orjson

from .metrics import METRIC_WEIGHTS


TYPE_TO_INT = {"clicks": 0, "carts": 1, "orders": 2}
INT_TO_TYPE = ("clicks", "carts", "orders")
# Target-sensitive matrix configurations.
MATRIX_WEIGHTS = np.asarray(
    [
        [1.0, 0.0, 0.0],  # clicks-only, lookback 2
        [1.0, 9.0, 1.0],  # cart-focused, lookback 2
        [1.0, 3.0, 6.0],  # conversion-focused, lookback 5
    ],
    dtype=np.float32,
)
MATRIX_LOOKBACK = np.asarray([2, 2, 5], dtype=np.int32)

FEATURE_NAMES = [
    "retrieval_score",
    "retrieval_reciprocal_rank",
    "repeat_score",
    "session_count",
    "last_position",
    "log_age_seconds",
    "last_type",
    "session_click_count",
    "session_cart_count",
    "session_order_count",
    "seen_as_click",
    "seen_as_cart",
    "seen_as_order",
    "is_last_item",
    "is_previous_item",
    "log_item_clicks",
    "log_item_carts",
    "log_item_orders",
    "item_cart_rate",
    "item_order_rate",
]
for _m in range(3):
    FEATURE_NAMES += [
        f"m{_m}_last_source_score",
        f"m{_m}_last_source_rank",
        f"m{_m}_sum",
        f"m{_m}_max",
        f"m{_m}_best_rank",
        f"m{_m}_hits",
        f"m{_m}_second_hop",
    ]


@dataclass
class ValidationRecord:
    session: int
    observed: list[tuple[int, int, int]]  # aid, ts, type
    truth: tuple[set[int], set[int], set[int]]


@nb.njit(cache=True)
def _heavy_hitter_update(
    neighbors: np.ndarray,
    scores: np.ndarray,
    matrix_idx: int,
    source: int,
    target: int,
    weight: float,
) -> None:
    if source == target or source < 0 or target < 0:
        return
    row_n = neighbors[matrix_idx, source]
    row_s = scores[matrix_idx, source]
    empty = -1
    min_idx = 0
    min_score = row_s[0]
    for k in range(row_n.shape[0]):
        if row_n[k] == target:
            row_s[k] += weight
            return
        if row_n[k] < 0 and empty < 0:
            empty = k
        if row_s[k] < min_score:
            min_score = row_s[k]
            min_idx = k
    if empty >= 0:
        row_n[empty] = target
        row_s[empty] = weight
    else:
        # Space-Saving update: bounded-memory approximation of per-source top-k.
        row_n[min_idx] = target
        row_s[min_idx] = min_score + weight


@nb.njit(cache=True)
def _update_batch(
    aids: np.ndarray,
    types: np.ndarray,
    offsets: np.ndarray,
    neighbors: np.ndarray,
    scores: np.ndarray,
    popularity: np.ndarray,
) -> None:
    n_matrices = MATRIX_LOOKBACK.shape[0]
    for s in range(offsets.shape[0] - 1):
        lo = offsets[s]
        hi = offsets[s + 1]
        for j in range(lo, hi):
            target = aids[j]
            target_type = types[j]
            if 0 <= target < popularity.shape[1]:
                popularity[target_type, target] += 1
            for m in range(n_matrices):
                weight = MATRIX_WEIGHTS[m, target_type]
                if weight <= 0:
                    continue
                start = max(lo, j - MATRIX_LOOKBACK[m])
                for i in range(start, j):
                    source = aids[i]
                    if 0 <= source < neighbors.shape[1] and 0 <= target < neighbors.shape[1]:
                        _heavy_hitter_update(neighbors, scores, m, source, target, weight)


def _stable_u64(value: int) -> int:
    x = value & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def _split_session(events: list[dict], session: int) -> tuple[list[dict], list[dict]]:
    if len(events) < 2:
        return events, []
    cut = 1 + _stable_u64(session) % (len(events) - 1)
    return events[:cut], events[cut:]


def _make_truth(future: list[dict]) -> tuple[set[int], set[int], set[int]]:
    clicks: set[int] = set()
    carts: set[int] = set()
    orders: set[int] = set()
    for e in future:
        typ = TYPE_TO_INT[e["type"]]
        aid = int(e["aid"])
        if typ == 0 and not clicks:
            clicks.add(aid)
        elif typ == 1:
            carts.add(aid)
        elif typ == 2:
            orders.add(aid)
    return clicks, carts, orders


def _as_compact_events(events: list[dict]) -> list[tuple[int, int, int]]:
    return [(int(e["aid"]), int(e["ts"]), TYPE_TO_INT[e["type"]]) for e in events]


def stream_fit_retrieval(
    train_path: str,
    max_aid: int,
    topk: int,
    validation_start_ms: int,
    validation_records: int,
    max_sessions: int,
    batch_events: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[ValidationRecord], dict]:
    n_items = max_aid + 1
    neighbors = np.full((3, n_items, topk), -1, dtype=np.int32)
    scores = np.zeros((3, n_items, topk), dtype=np.float32)
    popularity = np.zeros((3, n_items), dtype=np.int32)
    records: list[ValidationRecord] = []
    reservoir_seen = 0
    reservoir_rng = random.Random(20230201)
    batch_aids: list[int] = []
    batch_types: list[int] = []
    offsets = [0]
    stats = {
        "sessions": 0,
        "events": 0,
        "historical_sessions": 0,
        "pseudo_test_sessions": 0,
        "pseudo_test_eligible": 0,
        "events_trimmed_at_boundary": 0,
    }

    def flush() -> None:
        nonlocal batch_aids, batch_types, offsets
        if len(offsets) <= 1:
            return
        _update_batch(
            np.asarray(batch_aids, dtype=np.int32),
            np.asarray(batch_types, dtype=np.int8),
            np.asarray(offsets, dtype=np.int64),
            neighbors,
            scores,
            popularity,
        )
        batch_aids = []
        batch_types = []
        offsets = [0]

    started = time.time()
    with open(train_path, "rb") as f:
        for line_no, line in enumerate(f):
            if max_sessions and line_no >= max_sessions:
                break
            row = orjson.loads(line)
            session = int(row["session"])
            events = row["events"]
            stats["sessions"] += 1
            stats["events"] += len(events)
            # New sessions that start in the held-out period form the local
            # evaluation set. Older sessions remain historical but are
            # trimmed at the boundary to prevent future leakage.
            is_pseudo_test = bool(events and int(events[0]["ts"]) >= validation_start_ms)
            if is_pseudo_test:
                stats["pseudo_test_sessions"] += 1
                observed, future = _split_session(events, session)
                if future:
                    stats["pseudo_test_eligible"] += 1
                    reservoir_seen += 1
                    rec = ValidationRecord(session, _as_compact_events(observed), _make_truth(future))
                    if len(records) < validation_records:
                        records.append(rec)
                    else:
                        replace = reservoir_rng.randrange(reservoir_seen)
                        if replace < validation_records:
                            records[replace] = rec
                model_events = observed
            else:
                stats["historical_sessions"] += 1
                model_events = [e for e in events if int(e["ts"]) < validation_start_ms]
                stats["events_trimmed_at_boundary"] += len(events) - len(model_events)

            for e in model_events:
                batch_aids.append(int(e["aid"]))
                batch_types.append(TYPE_TO_INT[e["type"]])
            offsets.append(len(batch_aids))
            if len(batch_aids) >= batch_events:
                flush()

            if stats["sessions"] % 1_000_000 == 0:
                elapsed = time.time() - started
                print(
                    f"streamed {stats['sessions']:,} sessions / {stats['events']:,} events "
                    f"in {elapsed / 60:.1f} min; sampled={len(records):,}",
                    flush=True,
                )
    flush()
    stats["stream_seconds"] = time.time() - started
    return neighbors, scores, popularity, records, stats


def top_popularity(popularity: np.ndarray, n: int = 100) -> list[np.ndarray]:
    result = []
    for typ in range(3):
        row = popularity[typ]
        idx = np.argpartition(row, -n)[-n:]
        idx = idx[np.argsort(row[idx])[::-1]]
        result.append(idx.astype(np.int32))
    return result


def candidate_features(
    observed: list[tuple[int, int, int]],
    target_type: int,
    neighbors: np.ndarray,
    scores: np.ndarray,
    popularity: np.ndarray,
    popular_items: list[np.ndarray],
    max_candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not observed:
        aids = popular_items[target_type][:max_candidates]
        return aids, np.zeros((len(aids), len(FEATURE_NAMES)), dtype=np.float32)

    now = observed[-1][1]
    n_events = len(observed)
    per_item: dict[int, list[float]] = {}
    for pos, (aid, ts, typ) in enumerate(observed):
        if aid not in per_item:
            per_item[aid] = [0.0] * 8
        value = per_item[aid]
        value[0] += 1.0
        value[1] = float(pos)
        value[2] = float(ts)
        value[3] = float(typ)
        value[4 + typ] += 1.0

    matrix_maps: list[dict[int, list[float]]] = [dict(), dict(), dict()]
    second_hop_maps: list[dict[int, float]] = [dict(), dict(), dict()]
    last_source_maps: list[dict[int, tuple[float, float]]] = [dict(), dict(), dict()]
    prelim: dict[int, float] = {}
    repeat_scores: dict[int, float] = {}

    # Explicit repeat/revisit candidates.
    for reverse_pos, (aid, _ts, typ) in enumerate(reversed(observed[-50:])):
        type_boost = 1.0
        if target_type == 1 and typ == 1:
            type_boost = 2.5
        elif target_type == 2 and typ in (1, 2):
            type_boost = 3.0
        recency = math.exp(-reverse_pos / 10.0) * type_boost
        repeat_scores[aid] = max(repeat_scores.get(aid, 0.0), recency)
        prelim[aid] = prelim.get(aid, 0.0) + recency * (2.0 if target_type == 0 else 1.5)

    matrix_target_weights = (
        (1.0, 0.15, 0.30),
        (0.10, 1.0, 0.70),
        (0.05, 0.55, 1.0),
    )[target_type]

    # Diverse co-visitation candidates from the last unique session items.
    source_seen: set[int] = set()
    sources: list[tuple[int, int]] = []
    for reverse_pos, (aid, _ts, _typ) in enumerate(reversed(observed)):
        if aid not in source_seen:
            source_seen.add(aid)
            sources.append((aid, reverse_pos))
        if len(sources) == 30:
            break

    for source, reverse_pos in sources:
        source_weight = math.exp(-reverse_pos / 12.0)
        if source < 0 or source >= neighbors.shape[1]:
            continue
        for m in range(3):
            row_n = neighbors[m, source]
            row_s = scores[m, source]
            valid = row_n >= 0
            if not np.any(valid):
                continue
            ns = row_n[valid]
            ss = row_s[valid]
            order = np.argsort(ss)[::-1]
            total = float(ss.sum()) + 1e-8
            for rank0, idx in enumerate(order):
                aid = int(ns[idx])
                normalized = float(ss[idx]) / total
                contribution = source_weight * normalized
                values = matrix_maps[m].setdefault(aid, [0.0, 0.0, 1e9, 0.0])
                values[0] += contribution
                values[1] = max(values[1], contribution)
                values[2] = min(values[2], float(rank0 + 1))
                values[3] += 1.0
                prelim[aid] = prelim.get(aid, 0.0) + matrix_target_weights[m] * contribution * 20.0

    last_source = observed[-1][0]
    if 0 <= last_source < neighbors.shape[1]:
        for m in range(3):
            row_n = neighbors[m, last_source]
            row_s = scores[m, last_source]
            valid = row_n >= 0
            if not np.any(valid):
                continue
            ns = row_n[valid]
            ss = row_s[valid]
            total = float(ss.sum()) + 1e-8
            for rank0, idx in enumerate(np.argsort(ss)[::-1]):
                last_source_maps[m][int(ns[idx])] = (float(ss[idx]) / total, float(rank0 + 1))

    # Beam-like second hop. This captures alternatives that are not directly
    # connected to the observed item but are strongly connected through one
    # intermediate product.
    for m in range(3):
        beam = sorted(matrix_maps[m], key=lambda a: matrix_maps[m][a][0], reverse=True)[:20]
        for beam_rank, source in enumerate(beam):
            if source < 0 or source >= neighbors.shape[1]:
                continue
            row_n = neighbors[m, source]
            row_s = scores[m, source]
            valid = row_n >= 0
            if not np.any(valid):
                continue
            ns = row_n[valid]
            ss = row_s[valid]
            total = float(ss.sum()) + 1e-8
            beam_weight = matrix_maps[m][source][0] / (1.0 + 0.05 * beam_rank)
            for idx in np.argsort(ss)[::-1]:
                aid = int(ns[idx])
                contribution = 0.5 * beam_weight * float(ss[idx]) / total
                second_hop_maps[m][aid] = second_hop_maps[m].get(aid, 0.0) + contribution
                prelim[aid] = prelim.get(aid, 0.0) + matrix_target_weights[m] * contribution * 10.0

    # Popularity fallback is intentionally weak; it mainly fills short sessions.
    for rank0, aid_np in enumerate(popular_items[target_type][:50]):
        aid = int(aid_np)
        prelim[aid] = prelim.get(aid, 0.0) + 0.02 / (rank0 + 1)

    ranked = sorted(prelim, key=prelim.get, reverse=True)[:max_candidates]
    aids = np.asarray(ranked, dtype=np.int32)
    features = np.zeros((len(ranked), len(FEATURE_NAMES)), dtype=np.float32)
    previous_aid = observed[-2][0] if len(observed) > 1 else -1
    for row_idx, aid in enumerate(ranked):
        hist = per_item.get(aid, [0.0] * 8)
        item_clicks = float(popularity[0, aid]) if 0 <= aid < popularity.shape[1] else 0.0
        item_carts = float(popularity[1, aid]) if 0 <= aid < popularity.shape[1] else 0.0
        item_orders = float(popularity[2, aid]) if 0 <= aid < popularity.shape[1] else 0.0
        denom = item_clicks + item_carts + item_orders + 1.0
        values = [
            prelim.get(aid, 0.0),
            1.0 / (row_idx + 1.0),
            repeat_scores.get(aid, 0.0),
            hist[0],
            hist[1] / max(1.0, n_events - 1.0),
            math.log1p(max(0.0, (now - hist[2]) / 1000.0)) if hist[0] else 20.0,
            hist[3] if hist[0] else -1.0,
            hist[4],
            hist[5],
            hist[6],
            float(hist[4] > 0),
            float(hist[5] > 0),
            float(hist[6] > 0),
            float(aid == last_source),
            float(aid == previous_aid),
            math.log1p(item_clicks),
            math.log1p(item_carts),
            math.log1p(item_orders),
            item_carts / denom,
            item_orders / denom,
        ]
        for m in range(3):
            last_score, last_rank = last_source_maps[m].get(aid, (0.0, 0.0))
            values.extend([last_score, last_rank])
            mv = matrix_maps[m].get(aid, [0.0, 0.0, 0.0, 0.0])
            values.extend(mv)
            values.append(second_hop_maps[m].get(aid, 0.0))
        features[row_idx] = np.asarray(values, dtype=np.float32)
    return aids, features


def build_ranker_data(
    records: list[ValidationRecord],
    target_type: int,
    neighbors: np.ndarray,
    scores: np.ndarray,
    popularity: np.ndarray,
    popular_items: list[np.ndarray],
    max_candidates: int,
    train_fold: bool,
) -> tuple[np.ndarray, np.ndarray, list[int], dict]:
    selected = [r for r in records if (r.session % 5 != 0) == train_fold and r.truth[target_type]]
    capacity = max(1, len(selected) * max_candidates)
    x = np.empty((capacity, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.empty(capacity, dtype=np.int8)
    groups: list[int] = []
    cursor = 0
    positives_retrieved = 0
    positives_total = 0
    for i, rec in enumerate(selected):
        aids, feats = candidate_features(
            rec.observed,
            target_type,
            neighbors,
            scores,
            popularity,
            popular_items,
            max_candidates,
        )
        truth = rec.truth[target_type]
        labels = np.fromiter((int(int(a) in truth) for a in aids), dtype=np.int8, count=len(aids))
        n = len(aids)
        x[cursor : cursor + n] = feats
        y[cursor : cursor + n] = labels
        groups.append(n)
        cursor += n
        positives_retrieved += int(labels.sum())
        positives_total += min(20, len(truth))
        if (i + 1) % 5_000 == 0:
            print(f"  target={INT_TO_TYPE[target_type]} prepared {i + 1:,}/{len(selected):,} sessions", flush=True)
    stats = {
        "sessions": len(selected),
        "rows": cursor,
        "candidate_recall": positives_retrieved / positives_total if positives_total else 0.0,
    }
    return x[:cursor], y[:cursor], groups, stats


def train_ranker(
    x: np.ndarray,
    y: np.ndarray,
    groups: list[int],
    seed: int,
) -> lgb.LGBMRanker:
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=750,
        learning_rate=0.045,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=max(1, (os.cpu_count() or 4) - 1),
        verbosity=-1,
    )
    model.fit(x, y, group=groups, callbacks=[lgb.log_evaluation(100)])
    return model


def evaluate(
    records: list[ValidationRecord],
    models: list[lgb.LGBMRanker | None],
    neighbors: np.ndarray,
    scores: np.ndarray,
    popularity: np.ndarray,
    popular_items: list[np.ndarray],
    max_candidates: int,
) -> dict:
    hits_raw = np.zeros(3, dtype=np.int64)
    hits_ranked = np.zeros(3, dtype=np.int64)
    denominators = np.zeros(3, dtype=np.int64)
    candidate_hits = np.zeros(3, dtype=np.int64)
    evaluated = 0
    for rec in records:
        if rec.session % 5 != 0:
            continue
        evaluated += 1
        for target_type in range(3):
            truth = rec.truth[target_type]
            if not truth:
                continue
            aids, feats = candidate_features(
                rec.observed,
                target_type,
                neighbors,
                scores,
                popularity,
                popular_items,
                max_candidates,
            )
            denominator = min(20, len(truth))
            denominators[target_type] += denominator
            candidate_hits[target_type] += len(set(map(int, aids)) & truth)
            hits_raw[target_type] += len(set(map(int, aids[:20])) & truth)
            model = models[target_type]
            if model is None or len(aids) == 0:
                ranked = aids[:20]
            else:
                pred = model.predict(feats)
                ranked = aids[np.argsort(pred)[::-1][:20]]
            hits_ranked[target_type] += len(set(map(int, ranked)) & truth)
        if evaluated % 2_000 == 0:
            print(f"evaluated {evaluated:,} held-out sessions", flush=True)

    raw = np.divide(hits_raw, denominators, out=np.zeros(3, dtype=float), where=denominators > 0)
    ranked = np.divide(hits_ranked, denominators, out=np.zeros(3, dtype=float), where=denominators > 0)
    ceiling = np.divide(candidate_hits, denominators, out=np.zeros(3, dtype=float), where=denominators > 0)
    return {
        "sessions": evaluated,
        "denominators": denominators.tolist(),
        "candidate_recall": dict(zip(INT_TO_TYPE, ceiling.tolist())),
        "raw_retrieval": {
            **dict(zip(INT_TO_TYPE, raw.tolist())),
            "weighted": float(raw @ METRIC_WEIGHTS),
        },
        "ranked": {
            **dict(zip(INT_TO_TYPE, ranked.tolist())),
            "weighted": float(ranked @ METRIC_WEIGHTS),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--artifacts", default="artifacts/model")
    ap.add_argument("--max-aid", type=int, default=1_855_602)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max-candidates", type=int, default=120)
    ap.add_argument("--validation-start-ms", type=int, default=1_661_119_200_000)
    ap.add_argument("--validation-records", type=int, default=30_000)
    ap.add_argument("--max-sessions", type=int, default=0)
    ap.add_argument("--batch-events", type=int, default=1_000_000)
    ap.add_argument("--save-matrices", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()

    artifact_dir = Path(args.artifacts)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("Stage 1/4: streaming retrieval training", flush=True)
    neighbors, scores, popularity, records, stream_stats = stream_fit_retrieval(
        args.train,
        args.max_aid,
        args.topk,
        args.validation_start_ms,
        args.validation_records,
        args.max_sessions,
        args.batch_events,
    )
    print("stream_stats", stream_stats, flush=True)
    print(f"validation reservoir: {len(records):,}", flush=True)
    if args.retrieval_only:
        np.save(artifact_dir / "neighbors.npy", neighbors)
        np.save(artifact_dir / "scores.npy", scores)
        np.save(artifact_dir / "popularity.npy", popularity)
        with open(artifact_dir / "retrieval_stats.json", "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "stream_stats": stream_stats}, f, indent=2)
        print(f"Retrieval artifacts saved to {artifact_dir}", flush=True)
        return
    if len(records) < 100:
        raise RuntimeError("Too few pseudo-test records; check validation cutoff or max-sessions")

    popular_items = top_popularity(popularity)
    models: list[lgb.LGBMRanker | None] = [None, None, None]
    train_stats = {}
    print("Stage 2/4: candidate generation and ranker training", flush=True)
    for target_type in range(3):
        x, y, groups, stats = build_ranker_data(
            records,
            target_type,
            neighbors,
            scores,
            popularity,
            popular_items,
            args.max_candidates,
            train_fold=True,
        )
        train_stats[INT_TO_TYPE[target_type]] = stats
        print(f"ranker_data {INT_TO_TYPE[target_type]} {stats}", flush=True)
        if y.sum() == 0:
            print(f"No positives for {INT_TO_TYPE[target_type]}; skipping ranker", flush=True)
            continue
        models[target_type] = train_ranker(x, y, groups, 2023 + target_type)
        models[target_type].booster_.save_model(str(artifact_dir / f"ranker_{INT_TO_TYPE[target_type]}.txt"))
        del x, y

    print("Stage 3/4: exact Recall@20 evaluation", flush=True)
    metrics = evaluate(
        records,
        models,
        neighbors,
        scores,
        popularity,
        popular_items,
        args.max_candidates,
    )
    print(json.dumps(metrics, indent=2), flush=True)

    print("Stage 4/4: saving reproducibility artifacts", flush=True)
    report = {
        "config": vars(args),
        "feature_names": FEATURE_NAMES,
        "stream_stats": stream_stats,
        "ranker_train": train_stats,
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }
    with open(artifact_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    np.save(artifact_dir / "popularity.npy", popularity)
    with open(artifact_dir / "validation_sample.pkl", "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    if args.save_matrices:
        np.save(artifact_dir / "neighbors.npy", neighbors)
        np.save(artifact_dir / "scores.npy", scores)
    print(f"Complete in {(time.time() - started) / 60:.1f} minutes", flush=True)
    print(f"Report: {artifact_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
