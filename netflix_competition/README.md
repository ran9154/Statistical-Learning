# Netflix Class Competition – Statistical Learning (TAU 2026-7)

Predict users' ratings of **"Miss Congeniality" (2000)** from their ratings (and rating dates) of 99 other movies.
Evaluation metric: **RMSE** on 2,931 test users.

| Benchmark | RMSE |
|---|---|
| Predict mean | ≈ 0.957 |
| Simple linear regression (course baseline) | 0.7769 |
| Bonus threshold | **< 0.76** |
| Double-bonus threshold | **< 0.745** |

## Structure
```
netflix_competition/
├── data/              # raw data (git-ignored; downloaded automatically)
├── src/
│   └── data_loader.py # download + load data (0 → NaN), returns a NetflixData object
└── eda/
    ├── eda.py         # exploratory data analysis script
    ├── eda_output.txt # printed statistics
    ├── figures/       # plots
    └── README.md      # EDA findings & takeaways
```

## Quick start
```bash
pip install -r requirements.txt
python netflix_competition/eda/eda.py     # downloads data on first run
```

```python
from data_loader import load_data
d = load_data()     # d.X_train, d.X_test, d.D_train, d.D_test, d.y_train, d.y_date_train, d.y_date_test, d.movies
```

## Progress log
- [x] **Step 1 – EDA** → see [eda/README.md](eda/README.md)
- [ ] Step 2 – Baselines & CV framework
- [ ] Step 3 – Modeling
- [ ] Step 4 – Ensembling & submission
