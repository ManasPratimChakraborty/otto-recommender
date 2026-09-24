# OTTO Multi-Objective Recommender

A memory-bounded session recommender that predicts the next clicked, carted, and ordered products from event sequences. The complete pipeline streams the raw event data, builds target-sensitive item transitions, generates candidates, trains objective-specific rankers, and evaluates Recall@20.

## Performance

The final model achieved **0.568754 weighted Recall@20** on a leakage-safe temporal holdout of 19,982 sessions.

| Objective | Recall@20 |
|---|---:|
| Clicks | 0.540524 |
| Carts | 0.425687 |
| Orders | 0.644993 |
| **Weighted** | **0.568754** |

The combined metric is `0.10 × clicks + 0.30 × carts + 0.60 × orders`.

## Architecture

The model uses:

- three directed, target-sensitive co-visitation matrices;
- bounded Top-40 neighbors for each product;
- first-hop and second-hop sequential candidate retrieval;
- repeat, recency, event-type, transition, and popularity signals;
- up to 200 candidates per session and objective;
- 41 ranking features;
- separate 750-tree LightGBM LambdaRank models for clicks, carts, and orders.

All retrieval statistics and model parameters are learned from the local training data. The implementation is designed for CPU training with bounded memory.

## Project structure

```text
.
├── artifacts/                  # Generated model files (ignored by Git)
├── data/                       # Raw input files (ignored by Git)
├── src/
│   └── otto_recommender/
│       ├── __init__.py
│       ├── metrics.py
│       └── pipeline.py
├── tests/
│   ├── conftest.py
│   └── test_metrics.py
├── main.py                     # CLI entry point
├── requirements.txt
└── README.md
```

## Setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the raw files under `data/` as described in [data/README.md](data/README.md).

## Train and evaluate

Run the final configuration:

```bash
python main.py \
  --train data/train.jsonl \
  --validation-records 100000 \
  --topk 40 \
  --max-candidates 200 \
  --artifacts artifacts/model
```

On Windows PowerShell:

```powershell
python main.py `
  --train data\train.jsonl `
  --validation-records 100000 `
  --topk 40 `
  --max-candidates 200 `
  --artifacts artifacts\model
```

The pipeline performs four stages:

1. Streams sessions and trains bounded retrieval matrices.
2. Builds labeled candidate groups and trains three rankers.
3. Computes aggregate Recall@20 on the untouched temporal fold.
4. Saves rankers, metrics, and optional retrieval matrices.

Use `--save-matrices` to save retrieval arrays during a full training run. Use `--retrieval-only` when only retrieval artifacts are required.

## Tests

```bash
pytest -q
```

## Resource profile

The recorded full run processed 12,899,779 sessions and 216,716,096 events on a 14-core CPU machine with 31 GB RAM. The final local model package occupies approximately 1.7 GB, primarily from the two Top-40 retrieval arrays.
