# E-commerce Recommendation System

## Overview
Portfolio project for an e-commerce recommender on the RetailRocket dataset.
The pipeline turns raw clickstream events into implicit user–item interactions,
evaluates ranking metrics, and starts with a popularity baseline before more
advanced models (matrix factorization / neural CF) are added.

## Dataset
This project uses the RetailRocket dataset from Kaggle:
https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

1. Download the dataset manually from Kaggle.
2. Place `events.csv` into `data/raw/`.
3. Do not commit dataset files to this repository.

Expected columns: `timestamp` (ms), `visitorid`, `event`, `itemid`, `transactionid`.
Event types: `view`, `addtocart`, `transaction`.

## Project Structure
```text
.
├── .github/workflows/tests.yml
├── data/
│   ├── processed/          # train/test parquet, maps, stats.json
│   └── raw/                # events.csv (not committed)
├── notebooks/
├── results/                # baseline.json and future experiment outputs
├── scripts/
│   └── run_baseline.py
├── src/
│   ├── api/
│   ├── data/prepare.py
│   ├── evaluation/metrics.py
│   └── models/popularity.py
├── tests/
├── pytest.ini
└── requirements.txt
```

## Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage
Prepare interactions (filters sparse users/items, temporal 2-week split):

```bash
python -m src.data.prepare --input data/raw/events.csv --output data/processed
```

Evaluate the popularity baseline at K=10:

```bash
python scripts/run_baseline.py --data-dir data/processed --output results/baseline.json
```

Run tests:

```bash
pytest
```

## Results
| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Popularity | 0.0052 | 0.0088 | 0.0099 |

## Architecture
1. **Prepare** — map events to implicit weights (view=1, addtocart=3, transaction=5),
   aggregate per (user, item), iteratively drop users/items with fewer than 5
   interactions, re-index IDs, temporal split (last 2 weeks = test).
2. **Baseline** — global item popularity excluding items already seen in train.
3. **Evaluate** — Precision@K, Recall@K, NDCG@K macro-averaged over test users.
4. **Next** — learned models and a small serving API (planned).

## Tech Stack
- Python 3.11
- pandas
- numpy
- pyarrow
- pytest
- TensorFlow (planned)
- Flask (planned)
- Docker (planned)
