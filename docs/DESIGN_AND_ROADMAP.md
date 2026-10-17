# Design, Results, and Roadmap

This document explains what the current pipeline does, how the pieces fit
together, how to read the popularity baseline scores, and where to go next —
including concrete places to introduce TensorFlow.

---

## 1. Goal of the project

Build a portfolio-ready **e-commerce recommendation system** on the
[RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
clickstream:

- Turn raw events (`view` / `addtocart` / `transaction`) into implicit
  user–item feedback.
- Split data in a **time-aware** way so evaluation mimics production
  (train on the past, test on the future).
- Ship a **simple baseline** first (global popularity), with clean metrics and
  tests, so every later model has something honest to beat.
- Leave a clear path to learned models (matrix factorization, neural CF) and
  a small serving API.

---

## 2. What was implemented

### 2.1 Data preparation — `src/data/prepare.py`

| Step | What happens |
|------|----------------|
| Load | Read `data/raw/events.csv`, convert `timestamp` from milliseconds → UTC datetime. |
| Weight | Map events to implicit strengths: **view=1**, **addtocart=3**, **transaction=5**. |
| Aggregate | Sum weights per `(visitorid, itemid)`; keep the **latest** timestamp for that pair. |
| Filter | Iteratively drop users with &lt; 5 interactions and items with &lt; 5 interactions until the set is stable (removes cold-start noise). |
| Re-index | Map original IDs → contiguous `0…n-1` integers; save `user_map.parquet` / `item_map.parquet`. |
| Split | **Last 2 weeks** → test; everything earlier → train. Drop test rows whose user or item never appears in train. |
| Artifacts | Write `train.parquet`, `test.parquet`, `stats.json` under `data/processed/`. |

CLI example:

```bash
python -m src.data.prepare --input data/raw/events.csv --output data/processed
```

### 2.2 Metrics — `src/evaluation/metrics.py`

Pure NumPy ranking metrics with type hints and docstrings:

- **Precision@K** — fraction of the top-K list that is relevant.
- **Recall@K** — fraction of the user’s held-out items recovered in top-K.
- **NDCG@K** — discounted ranking quality (binary relevance), normalized by the ideal DCG.

`evaluate(recommend_fn, train_df, test_df, k=10)`:

1. For each test user, builds the ground-truth item set.
2. Asks the model for recommendations.
3. **Excludes items already seen in train** (no credit for recommending the past).
4. Macro-averages metrics over test users.

### 2.3 Popularity baseline — `src/models/popularity.py`

- `fit(train_df)`: score each item by sum of implicit weights (fallback: counts).
- `recommend(user_id, k)`: return the top-K global items **excluding** that user’s train history.

This is intentionally dumb and strong: many real catalogs have a heavy head of
popular products, so popularity is a tough first bar.

### 2.4 Runner, tests, CI

- `scripts/run_baseline.py` loads processed data, evaluates at K=10, writes
  `results/baseline.json`.
- `tests/test_metrics.py` — hand-computed Precision / Recall / NDCG cases.
- `tests/test_prepare.py` — tiny synthetic frame covering weights, filtering, split.
- GitHub Actions still runs `pytest` on push/PR (`Python 3.11` in CI).

---

## 3. Architecture (current)

```text
┌─────────────────────┐
│  data/raw/events.csv│  RetailRocket clickstream
└──────────┬──────────┘
           │  prepare.py
           ▼
┌──────────────────────────────────────────────┐
│  data/processed/                             │
│  train.parquet · test.parquet                │
│  user_map.parquet · item_map.parquet         │
│  stats.json                                  │
└──────────┬───────────────────────┬───────────┘
           │                       │
           ▼                       ▼
┌─────────────────────┐   ┌────────────────────┐
│ PopularityRecommender│   │  (future models)   │
│ fit → recommend      │   │  MF / NeuMF / …    │
└──────────┬──────────┘   └─────────┬──────────┘
           │                        │
           └────────────┬───────────┘
                        ▼
           ┌────────────────────────┐
           │ evaluation/metrics.py  │
           │ Precision · Recall ·   │
           │ NDCG @ K               │
           └────────────┬───────────┘
                        ▼
           ┌────────────────────────┐
           │ results/*.json         │
           │ (+ notebooks / API)    │
           └────────────────────────┘
```

**Data flow idea:** raw events → dense-enough interaction matrix → temporal
holdout → model → ranked list → offline ranking metrics.

**Serving (planned, not built yet):** `src/api/` will eventually wrap a fitted
model behind a small HTTP endpoint (`GET /recommend?user_id=&k=`), with the
same “exclude seen items” rule used in evaluation.

---

## 4. Dataset stats after preparation

From `data/processed/stats.json` (min interactions = 5, test = 2 weeks):

| Stat | Value |
|------|------:|
| Users (in train catalog) | 21,664 |
| Items (in train catalog) | 18,239 |
| Train interactions | 229,933 |
| Test interactions | 8,181 |
| Total interactions (train+test) | 238,114 |
| Sparsity | ~99.94% |
| Train window | 2015-05-03 → 2015-09-04 |
| Test window | 2015-09-04 → 2015-09-18 |

Notes:

- Raw RetailRocket has ~2.76M events and ~1.4M visitors. After aggregation and
  iterative filtering, we keep a much smaller but **denser** subset that
  collaborative filtering can actually learn from.
- Sparsity near 100% is normal for e-commerce; it is why naive dense matrices
  do not scale and why we will prefer sparse / embedding approaches later.
- Only **1,532 users** appear in the cleaned test set used for metrics (many
  users have no valid future interactions after cold-start drops). That is
  expected with a short holdout and strict train-overlap rules.

---

## 5. Baseline results — how to read them

From `results/baseline.json` (K = 10):

| Metric | Score |
|--------|------:|
| Precision@10 | **0.0052** |
| Recall@10 | **0.0088** |
| NDCG@10 | **0.0099** |

### Are these “bad”?

They look small in absolute terms, but that is **typical** for this setup:

1. **Huge catalog** (~18k items). Random guessing into top-10 would score near
   `10 / 18239 ≈ 0.00055` Precision@10. Popularity is already roughly
   **~10× random** on precision — useful, not useless.
2. **Sparse, short test histories.** Many users have only one held-out item.
   Missing that single item → Precision@10 = 0 for that user. Macro-averaging
   pulls the mean down hard.
3. **Popularity ignores personalization.** It recommends the same head items to
   everyone (minus seen items). Users with niche tastes get almost no help.
4. **Temporal split is stricter than random split.** Models cannot “cheat” by
   peeking at future co-occurrences; scores are usually lower than in many
   blog posts that use random holdouts.

### Verdict

The baseline is doing its job: it establishes a **credible floor**. Any learned
model that does not beat these numbers on the **same** split and metrics is not
worth deploying. For a portfolio, the story is: “I measured a fair temporal
baseline; next models must improve Recall/NDCG@10 under the same protocol.”

---

## 6. Suggestions to advance further

Ordered from highest leverage / lowest complexity:

### Phase A — Stronger classical recommenders (still no deep learning)

1. **Item–item kNN (cosine / Jaccard on implicit weights)**  
   Fast to implement, often beats popularity on RetailRocket-style data.
2. **Implicit ALS / WRMF** (e.g. `implicit` library)  
   Classic matrix factorization for confidence-weighted implicit feedback.
   Use the same train/test files and `evaluate()`.
3. **Simple hybrids**  
   `score = α · popularity + (1-α) · CF` to cover cold and warm users.
4. **Better evaluation protocol**  
   - Report metrics with confidence intervals (bootstrap over users).  
   - Add coverage / novelty / catalog diversity so you do not overfit to head
     items.  
   - Optional: leave-one-out per user vs current global time cut (document the
     tradeoff).

### Phase B — Features and cold-start

5. Use RetailRocket **item properties** and **category tree** for content
   signals (especially new items with few interactions).
6. Session / sequential features: last-N views, time since last event, funnel
   stage (`view` → `cart` → `buy`).

### Phase C — Neural models (TensorFlow) and serving

7. Neural Collaborative Filtering / two-tower retrieval (see next section).
8. Thin **Flask/FastAPI** service loading embeddings or a popular+CF hybrid.
9. Docker image + a short demo notebook or Streamlit UI for recruiters.

### Engineering hygiene worth doing soon

- Pin dependency versions; keep CI on 3.11 while local may be newer.
- Add a `Makefile` or `just` recipes: `prepare`, `baseline`, `test`.
- Log experiment configs next to `results/*.json` (seed, K, min interactions).

---

## 7. Where TensorFlow fits

TensorFlow is **not** needed for the popularity baseline. Use it when you want
**learned representations** or **non-linear ranking** on the same processed
tables.

### Best insertion points in *this* repo

| Location | Role |
|----------|------|
| `src/models/ncf.py` (new) | Neural Collaborative Filtering: user & item embedding tables + MLP on the concatenation (NeuMF-style). Train on `(user_id, item_id, weight)` from `train.parquet` with negative sampling. |
| `src/models/two_tower.py` (new) | Two-tower retrieval: separate user/item encoders, dot-product score. Scale to larger catalogs via ANN (optional later). |
| `src/models/sequential.py` (later) | Transformer / GRU over recent item IDs in a session (next-item prediction). Needs event-level sequences, not only aggregated pairs — extend `prepare.py` to also emit sequences. |
| `scripts/train_tf.py` (new) | Training loop: `tf.data` pipeline from parquet → batched (user, positive item, negatives) → Keras model → checkpoints under `/models/` (gitignored). |
| `scripts/run_ncf.py` (new) | Mirror `run_baseline.py`: load weights, wrap `recommend(user_id, k)`, call `evaluate()`, write `results/ncf.json`. |
| `src/api/` (later) | Load SavedModel / embedding matrices and serve top-K; keep metric-compatible filtering of seen items. |

### Recommended first TensorFlow experiment (practical)

1. Keep **exactly** the current `train.parquet` / `test.parquet` and metrics.
2. Train a small **NeuMF**:
   - Embedding dim 32–64.
   - Binary cross-entropy with 4–20 negatives per positive.
   - Sample probability by popularity (to avoid easy negatives).
3. At inference, score all items (or a candidate set) for a user, remove train
   items, take top-10.
4. Compare `results/ncf.json` vs `results/baseline.json` on Precision / Recall /
   NDCG@10.

If NeuMF does not beat ALS/item-kNN by a clear margin, say so in the README —
that honesty is a feature in a portfolio, not a bug.

### What TensorFlow should *not* replace yet

- Data prep and temporal split (stay in pandas).
- Metric definitions (stay in NumPy so tests remain simple and framework-free).
- The popularity baseline (keep as the permanent reference line).

---

## 8. How to reproduce

```bash
# 1. Place events.csv in data/raw/ (Kaggle or HF mirror)
pip install -r requirements.txt

# 2. Prepare
python -m src.data.prepare --input data/raw/events.csv --output data/processed

# 3. Baseline
python scripts/run_baseline.py --data-dir data/processed --output results/baseline.json

# 4. Tests
pytest
```

---

## 9. Summary

| Piece | Status |
|-------|--------|
| Event → implicit interactions + temporal split | Done |
| Ranking metrics + train-item exclusion | Done |
| Popularity baseline + saved scores | Done |
| Unit tests + CI hook | Done |
| Item-kNN / ALS | Next classical step |
| TensorFlow NeuMF / two-tower | Next deep step (`src/models/` + train script) |
| API + Docker | After a model beats the baseline |

The current scores are a **fair, low personalized floor** on a sparse temporal
split. The architecture is ready to plug in stronger models without changing
how success is measured — that consistency is what makes later TensorFlow work
meaningful.
