# Model artifacts

Training writes retrieval matrices, popularity statistics, rankers, validation samples, and metric reports into this directory.

Generated artifacts are excluded from version control. A complete run normally produces:

```text
artifacts/model/
├── neighbors.npy
├── scores.npy
├── popularity.npy
├── ranker_clicks.txt
├── ranker_carts.txt
├── ranker_orders.txt
├── report.json
├── retrieval_stats.json
└── validation_sample.pkl
```
