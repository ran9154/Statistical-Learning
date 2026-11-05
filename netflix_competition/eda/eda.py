"""
eda.py -- Exploratory Data Analysis for the class Netflix competition
======================================================================

Run from the repo root:
    python netflix_competition/eda/eda.py

Outputs:
    netflix_competition/eda/figures/*.png   -- all plots
    netflix_competition/eda/eda_output.txt  -- all printed statistics

Questions we try to answer (the "points to ponder" from the assignment):
  1. What do the data look like (shapes, consistency of ratings vs. dates)?
  2. How is the target ("Miss Congeniality") distributed? What's the naive RMSE?
  3. How much is missing, and is the *missingness itself* informative?
  4. Which movies are most correlated with the target?
  5. Is a user's general rating level (user mean) predictive?
  6. Do dates matter (rating date of the target, time between ratings, etc.)?
  7. Does movie release year matter?
  8. Are the train and test sets similarly distributed?
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_loader import N_FULL, DATE_ORIGIN, load_data  # noqa: E402

FIG_DIR = ROOT / "eda" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = ROOT / "eda" / "eda_output.txt"
sns.set_theme(style="whitegrid")
_out_lines: list[str] = []


def log(*args) -> None:
    """Print and also keep the line for eda_output.txt."""
    s = " ".join(str(a) for a in args)
    print(s)
    _out_lines.append(s)


def section(title: str) -> None:
    log("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=110)
    plt.close()


def short(t: str, n: int = 32) -> str:
    return t if len(t) <= n else t[: n - 1] + "…"


# ---------------------------------------------------------------------------
d = load_data()
Xtr, Xte, Dtr, Dte = d.X_train, d.X_test, d.D_train, d.D_test
y, ydt, yde = d.y_train, d.y_date_train, d.y_date_test
movies = d.movies
pd.set_option("display.width", 160, "display.max_rows", 120)

# ---------------------------------------------------------------------------
section("1. Shapes & sanity checks")
log("X_train", Xtr.shape, " X_test", Xte.shape, " y_train", y.shape,
    " y_date_train", ydt.shape, " y_date_test", yde.shape)
log("Movie release years:", movies.year.min(), "-", movies.year.max())
log("Missing ratings in first 14 movies (train/test):",
    int(Xtr.iloc[:, :N_FULL].isna().sum().sum()), "/",
    int(Xte.iloc[:, :N_FULL].isna().sum().sum()))
log("Rating missing <=> date missing? train:",
    bool((Xtr.isna() == Dtr.isna()).all().all()),
    " test:", bool((Xte.isna() == Dte.isna()).all().all()))
log("Missing target ratings / dates:", int(y.isna().sum()), int(ydt.isna().sum()),
    int(yde.isna().sum()))
log("Rating values present:", sorted(np.unique(Xtr.values[~np.isnan(Xtr.values)])))
log("\nThe 14 fully-observed movies:")
log(movies.iloc[:N_FULL].to_string())

# ---------------------------------------------------------------------------
section("2. Target: rating of 'Miss Congeniality'")
vc = y.value_counts().sort_index()
log(pd.DataFrame({"count": vc, "pct": (100 * vc / len(y)).round(2)}).to_string())
log(f"mean={y.mean():.4f}  std={y.std(ddof=0):.4f}")
log(f"Naive RMSE (predict the train mean for everyone) = {y.std(ddof=0):.4f}")
log("Benchmarks: linear regression=0.7769, bonus<0.76, double bonus<0.745")

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=y.astype(int), ax=ax, color="steelblue")
ax.set(title="Miss Congeniality rating (train)", xlabel="rating", ylabel="users")
savefig("01_target_distribution.png")

# ---------------------------------------------------------------------------
section("3. Missingness")
miss = pd.DataFrame({
    "year": movies.year.values,
    "miss_train": Xtr.isna().mean().values,
    "miss_test": Xte.isna().mean().values,
}, index=movies.title)
log("Fraction missing per movie (sorted, most-missing first, top 15):")
log(miss.sort_values("miss_train", ascending=False).head(15).round(3).to_string())
n_rated_tr = Xtr.notna().sum(axis=1)
n_rated_te = Xte.notna().sum(axis=1)
log(f"\n#movies rated per user: train mean={n_rated_tr.mean():.1f} "
    f"median={n_rated_tr.median():.0f} min={n_rated_tr.min()} max={n_rated_tr.max()}")
log(f"                        test  mean={n_rated_te.mean():.1f} "
    f"median={n_rated_te.median():.0f} min={n_rated_te.min()} max={n_rated_te.max()}")
log("Overall missing rate (85 optional movies): train "
    f"{Xtr.iloc[:, N_FULL:].isna().mean().mean():.3f}, "
    f"test {Xte.iloc[:, N_FULL:].isna().mean().mean():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar(range(99), miss.miss_train.values, color="indianred")
axes[0].axvline(N_FULL - 0.5, color="k", ls="--")
axes[0].set(title="Fraction missing per movie (train)", xlabel="movie index",
            ylabel="fraction missing")
bins = np.arange(n_rated_tr.min(), 101)
axes[1].hist(n_rated_tr, bins=bins, density=True, alpha=.6, label="train")
axes[1].hist(n_rated_te, bins=bins, density=True, alpha=.6, label="test")
axes[1].set(title="# movies rated per user", xlabel="# rated (out of 99)")
axes[1].legend()
savefig("02_missingness.png")

# Is missingness informative about the target?
log("\nIs *missingness* informative? Mean target when movie is rated vs. missing")
rows = []
for j, t in enumerate(movies.title):
    m = Xtr.iloc[:, j].isna()
    if 0 < m.sum() < len(m):
        rows.append((t, m.mean(), y[~m].mean(), y[m].mean(),
                     np.corrcoef(m.astype(float), y)[0, 1]))
mi = pd.DataFrame(rows, columns=["title", "frac_missing", "y|rated", "y|missing",
                                 "corr(missing,y)"]).set_index("title")
mi["abs"] = mi["corr(missing,y)"].abs()
log(mi.sort_values("abs", ascending=False).drop(columns="abs").head(15).round(3).to_string())
log(f"corr(#movies rated, y) = {np.corrcoef(n_rated_tr, y)[0, 1]:.4f}")

fig, ax = plt.subplots(figsize=(6, 4))
grp = pd.qcut(n_rated_tr, 10, duplicates="drop")
y.groupby(grp, observed=True).mean().plot(marker="o", ax=ax)
ax.set(title="Mean target vs. # movies rated (deciles)", xlabel="# rated bin",
       ylabel="mean Miss Congeniality rating")
plt.xticks(rotation=45, ha="right")
savefig("03_target_vs_n_rated.png")

# ---------------------------------------------------------------------------
section("4. Movie-level stats & correlation with the target")
corr_rows = []
for j, t in enumerate(movies.title):
    x = Xtr.iloc[:, j]
    m = x.notna()
    corr_rows.append((t, movies.year[j], m.sum(), x.mean(), x.std(),
                      np.corrcoef(x[m], y[m])[0, 1]))
mc = pd.DataFrame(corr_rows, columns=["title", "year", "n_rated", "mean", "std",
                                      "corr_with_y"]).set_index("title")
log("Top 20 movies by correlation with the target (computed on users who rated both):")
log(mc.sort_values("corr_with_y", ascending=False).head(20).round(3).to_string())
log("\nBottom 5:")
log(mc.sort_values("corr_with_y").head(5).round(3).to_string())
log("\nFirst 14 (fully observed) movies:")
log(mc.iloc[:N_FULL].round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 12))
top = mc.sort_values("corr_with_y", ascending=True).tail(40)
ax.barh([short(t) for t in top.index], top.corr_with_y,
        color=["darkorange" if t in movies.title[:N_FULL].values else "steelblue"
               for t in top.index])
ax.set(title="Top-40 movies by corr. with Miss Congeniality\n(orange = fully observed)",
       xlabel="Pearson correlation")
savefig("04_corr_with_target.png")

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(pd.concat([Xtr.iloc[:, :N_FULL], y.rename("MISS CONGENIALITY")], axis=1)
            .rename(columns=short).corr(), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Correlation: 14 fully-observed movies + target")
savefig("05_corr_heatmap_first14.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(mc["mean"], mc["corr_with_y"], s=mc["n_rated"] / 200, alpha=.6)
ax.set(xlabel="movie mean rating", ylabel="corr with target",
       title="Movie mean vs corr with target (size = #ratings)")
savefig("06_movie_mean_vs_corr.png")

# ---------------------------------------------------------------------------
section("5. User-level features")
user_mean = Xtr.mean(axis=1)
user_mean14 = Xtr.iloc[:, :N_FULL].mean(axis=1)
user_std = Xtr.std(axis=1)
for name, f in [("user mean (all rated)", user_mean),
                ("user mean (first 14)", user_mean14),
                ("user std", user_std)]:
    log(f"corr({name}, y) = {np.corrcoef(f, y)[0, 1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(x=y.astype(int), y=user_mean, ax=axes[0], color="lightsteelblue")
axes[0].set(title="User mean rating by target rating", xlabel="Miss Congeniality",
            ylabel="user mean over rated movies")
ct = pd.crosstab(pd.qcut(user_mean, 5, labels=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"]),
                 y.astype(int), normalize="index")
ct.plot(kind="bar", stacked=True, ax=axes[1], colormap="RdYlGn")
axes[1].set(title="Target distribution by user-mean quintile", xlabel="",
            ylabel="fraction")
axes[1].legend(title="y", bbox_to_anchor=(1.02, 1), loc="upper left")
axes[1].tick_params(axis="x", rotation=0)
savefig("07_user_mean_vs_target.png")

# ---------------------------------------------------------------------------
section("6. Dates")
def to_dt(s):
    return DATE_ORIGIN + pd.to_timedelta(s, unit="D")

log("Target-rating date (train):", to_dt(ydt.min()).date(), "->", to_dt(ydt.max()).date())
log("Target-rating date (test): ", to_dt(yde.min()).date(), "->", to_dt(yde.max()).date())
log("All movie-rating dates (train):", to_dt(np.nanmin(Dtr.values)).date(), "->",
    to_dt(np.nanmax(Dtr.values)).date())
log(f"corr(target date, y) = {np.corrcoef(ydt, y)[0, 1]:.4f}")

first_date = Dtr.min(axis=1)
last_date = Dtr.max(axis=1)
feat = pd.DataFrame({
    "y_date": ydt,
    "days_since_first_rating": ydt - first_date,
    "days_until_last_rating": last_date - ydt,
    "user_activity_span": last_date - first_date,
    "n_same_day_as_y": (Dtr.values == ydt.values[:, None]).sum(axis=1),
    "frac_rated_before_y": (Dtr.values < ydt.values[:, None]).sum(axis=1) / n_rated_tr,
})
for c in feat:
    log(f"corr({c}, y) = {np.corrcoef(feat[c], y)[0, 1]:.4f}   "
        f"(median={feat[c].median():.1f})")
log(f"Fraction of users who rated Miss Congeniality on the SAME day as >=1 other "
    f"movie: {(feat.n_same_day_as_y > 0).mean():.3f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].hist(to_dt(ydt), bins=50, alpha=.6, density=True, label="train")
axes[0].hist(to_dt(yde), bins=50, alpha=.6, density=True, label="test")
axes[0].set(title="Date of Miss Congeniality rating")
axes[0].legend()
q = pd.qcut(ydt, 10)
y.groupby(q, observed=True).mean().plot(marker="o", ax=axes[1])
axes[1].set(title="Mean target vs target-date decile", xlabel="")
axes[1].tick_params(axis="x", rotation=45)
q2 = pd.qcut(feat.n_same_day_as_y.clip(upper=30), 8, duplicates="drop")
y.groupby(q2, observed=True).mean().plot(marker="o", ax=axes[2])
axes[2].set(title="Mean target vs #movies rated same day", xlabel="")
axes[2].tick_params(axis="x", rotation=45)
savefig("08_dates.png")

# Movie-specific: does a rating given on the same day as the target correlate more?
log("\nFor the top-5 correlated movies: corr with y when rated same day vs other days")
for t in mc.sort_values("corr_with_y", ascending=False).index[:5]:
    x, dd = Xtr[t], Dtr[t]
    same = (dd == ydt) & x.notna()
    other = (dd != ydt) & x.notna()
    log(f"  {short(t, 40):40s} same-day n={same.sum():5d} r={np.corrcoef(x[same], y[same])[0,1]:.3f}"
        f" | other n={other.sum():5d} r={np.corrcoef(x[other], y[other])[0,1]:.3f}")

# ---------------------------------------------------------------------------
section("7. Movie release year")
log(f"corr(release year, movie mean rating) = {np.corrcoef(mc.year, mc['mean'])[0, 1]:.4f}")
log(f"corr(release year, corr_with_y)       = {np.corrcoef(mc.year, mc.corr_with_y)[0, 1]:.4f}")
log(f"corr(release year, fraction missing)  = {np.corrcoef(mc.year, miss.miss_train)[0, 1]:.4f}")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(mc.year, mc["mean"]); axes[0].set(xlabel="release year", ylabel="mean rating")
axes[1].scatter(mc.year, mc.corr_with_y); axes[1].set(xlabel="release year", ylabel="corr with target")
axes[2].scatter(mc.year, miss.miss_train); axes[2].set(xlabel="release year", ylabel="fraction missing")
fig.suptitle("Movie release year (Miss Congeniality = 2000)")
savefig("09_release_year.png")

# ---------------------------------------------------------------------------
section("8. Train vs. test comparison")
cmp = pd.DataFrame({
    "train_mean_rating": Xtr.mean(), "test_mean_rating": Xte.mean(),
    "train_miss": Xtr.isna().mean(), "test_miss": Xte.isna().mean(),
})
log(f"Max |diff| in per-movie mean rating:  {(cmp.train_mean_rating - cmp.test_mean_rating).abs().max():.3f}")
log(f"Max |diff| in per-movie missing rate: {(cmp.train_miss - cmp.test_miss).abs().max():.3f}")
log(f"User-mean rating: train {user_mean.mean():.3f}  test {Xte.mean(axis=1).mean():.3f}")
log(f"Target date mean: train {ydt.mean():.1f}  test {yde.mean():.1f}")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(cmp.train_mean_rating, cmp.test_mean_rating)
axes[0].plot([1, 5], [1, 5], "k--"); axes[0].set(xlabel="train", ylabel="test",
                                                 title="Per-movie mean rating")
axes[1].scatter(cmp.train_miss, cmp.test_miss)
axes[1].plot([0, 1], [0, 1], "k--"); axes[1].set(xlabel="train", ylabel="test",
                                                 title="Per-movie missing rate")
savefig("10_train_vs_test.png")

OUT_FILE.write_text("\n".join(_out_lines), encoding="utf-8")
print(f"\nSaved figures to {FIG_DIR} and stats to {OUT_FILE}")
