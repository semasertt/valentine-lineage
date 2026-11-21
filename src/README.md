# src

| Module | Milestone | Content |
|---|---|---|
| `lineage_data.py` | 2 | load the benchmark CSVs and the lineage log |
| `eda.py` | 2 | value containment, name similarity, ground truth, pair labels |
| `preprocessing.py` | 3 | cleaning, candidate pairs, table-name variants |
| `matching.py` | 4 | run the Valentine matchers on all candidate pairs (cached in `results/`) |
| `evaluation.py` | 4 | column → table scores, ranking and threshold metrics, baselines, error analysis |
