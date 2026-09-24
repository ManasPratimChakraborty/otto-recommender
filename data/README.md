# Data directory

Place the OTTO files here before training:

```text
data/
├── train.jsonl
├── test.jsonl
└── sample_submission.csv
```

The raw files are intentionally excluded from version control because of their size. The training pipeline reads `train.jsonl` as a binary stream and does not load the complete event table into memory.
