"""
XGBoost model for maize storage weight loss (lowland tropical CIMMYT, 152 rows).
Deji - Final Year Project. Run: python build_xgboost.py --data cimmyt_lowland_tropical_merged.csv
"""
import argparse, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, spearmanr
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import shap
warnings.filterwarnings("ignore")

TARGET = "Weight_Loss  (%)"
SEEDS = [0, 1, 2, 3, 4]
HERMETIC = {"HBT", "HBZ", "HMS", "PBO", "PBA", "SPB"}
# End-of-storage measurements: co-measured with the target -> leakage. Never used.
LEAK = ["GMCF  (%)", "GTF  (°C)", "InsF  (%)", "FunF  (%)", "TdamF  (%)", "LMWF", "LLGBF", "AGMF"]
INTAKE = ["Elevation (m asl)", "Tmin (°C)", "Tmax (°C)", "Hrmin  (%)", "Hrmax  (%)",
          "Storage time (days)", "GMCI  (%)", "GTI  (°C)", "Grain Impurities (%)",
          "InsI  (%)", "FunI  (%)", "TdamI  (%)", "LMWI", "LLGBI"]  # AGMI dropped: constant


def emc_wb(T, RH):
    """Modified Henderson EMC for yellow dent corn (ASABE D245.6), wet basis %."""
    K, N, C = 8.6541e-5, 1.8634, 49.810
    rh = np.clip(RH / 100.0, 0.01, 0.99)
    mdb = (-np.log(1 - rh) / (K * (T + C))) ** (1 / N)
    return 100 * mdb / (100 + mdb)


def build_features(df):
    X = df[INTAKE].copy()
    X["native_variety"] = (df["Variety"] == "Native").astype(int)
    tech = df["Storage technology"].str.strip()
    X["hermetic"] = tech.isin(HERMETIC).astype(int)
    X = X.join(pd.get_dummies(tech, prefix="tech").astype(int))
    # Physics-derived features (Chapter 2 mechanisms)
    t_mean = (df["Tmin (°C)"] + df["Tmax (°C)"]) / 2
    rh_mean = (df["Hrmin  (%)"] + df["Hrmax  (%)"]) / 2
    days = df["Storage time (days)"]
    X["T_mean"] = t_mean
    X["RH_mean"] = rh_mean
    X["EMC_wb"] = emc_wb(t_mean, rh_mean)
    X["EMC_gap"] = df["GMCI  (%)"] - X["EMC_wb"]            # >0: grain wetter than equilibrium
    X["degree_days_15"] = np.clip(t_mean - 15, 0, None) * days  # insect development index
    X["humid_days_70"] = np.clip(rh_mean - 70, 0, None) * days  # moisture-pressure index
    X["exposure_open"] = days * (1 - X["hermetic"])          # time in non-hermetic storage
    assert not set(LEAK) & set(X.columns)
    return X


def models(seed, log_target):
    xgb = XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=3,
                       subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
                       gamma=0.0, tree_method="exact", n_jobs=1, random_state=seed)
    gbm = GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, max_depth=3,
                                    subsample=0.8, min_samples_leaf=3, random_state=seed)
    return {"XGBoost": xgb, "GBM (Friedman)": gbm,
            "OLS": make_pipeline(StandardScaler(), LinearRegression()),
            "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            "Mean predictor": None}


def oof_predict(X, y, splits, seed, log_target):
    out = {}
    for name, est in models(seed, log_target).items():
        pred = np.zeros(len(y))
        for tr, te in splits:
            if est is None:
                pred[te] = y.iloc[tr].mean(); continue
            yt = np.log1p(y.iloc[tr]) if log_target else y.iloc[tr]
            est.fit(X.iloc[tr], yt)
            p = est.predict(X.iloc[te])
            pred[te] = np.clip(np.expm1(p) if log_target else p, 0, None)
        out[name] = pred
    return out


def metrics(y, p):
    rmse = mean_squared_error(y, p) ** 0.5; mae = mean_absolute_error(y, p)
    return {"RMSE": rmse, "MAE": mae, "R2": r2_score(y, p), "Bias(obs-pred)": float(np.mean(y - p)),
            "RMSE/MAE": rmse / mae}


def grouped_splits(groups, seed, k=5):
    ug = np.array(sorted(groups.unique())); rng = np.random.RandomState(seed); rng.shuffle(ug)
    fold_of = {g: i % k for i, g in enumerate(ug)}
    f = groups.map(fold_of).values
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out, run = {}, 0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p)); out[k] = run
    return out


def main(path, outdir):
    df = pd.read_csv(path)
    X, y = build_features(df), df[TARGET]
    cond, site = df["condition_id"], df["site_id"]

    # Noise floor: replicate spread within identical conditions
    sd = df.groupby("condition_id")[TARGET].std(ddof=1).dropna()
    n = df.groupby("condition_id")[TARGET].count()
    pooled = np.sqrt(((n[sd.index] - 1) * sd ** 2).sum() / (n[sd.index] - 1).sum())
    r2_ceiling = 1 - pooled ** 2 / y.var(ddof=0)
    report = {"n_rows": len(df), "n_conditions": int(cond.nunique()), "n_sites": int(site.nunique()),
              "noise_floor_RMSE": pooled, "max_attainable_R2": r2_ceiling, "features": list(X.columns)}

    # 1. Target transform choice (grouped CV, seed 0) - decided on RMSE
    choice = {}
    for lt in (False, True):
        o = oof_predict(X, y, grouped_splits(cond, 0), 0, lt)
        choice["log1p" if lt else "raw"] = metrics(y, o["XGBoost"])["RMSE"]
    log_target = choice["log1p"] < choice["raw"]
    report["target_transform_RMSE"] = choice; report["log_target_used"] = bool(log_target)

    # 2. Primary: condition-grouped 5-fold CV, repeated over 5 seeds
    per_seed, oof0 = {}, None
    for s in SEEDS:
        o = oof_predict(X, y, grouped_splits(cond, s), s, log_target)
        if s == 0: oof0 = o
        for k, p in o.items(): per_seed.setdefault(k, []).append(metrics(y, p))
    summ = {k: {m: (np.mean([d[m] for d in v]), np.std([d[m] for d in v])) for m in v[0]}
            for k, v in per_seed.items()}
    report["grouped_cv"] = summ

    # Bootstrap 95% CI on RMSE (seed-0 OOF), 1000 resamples
    rng = np.random.RandomState(42); ci = {}
    for k, p in oof0.items():
        b = [mean_squared_error(y.iloc[i], p[i]) ** 0.5 for i in
             (rng.randint(0, len(y), len(y)) for _ in range(1000))]
        ci[k] = (np.percentile(b, 2.5), np.percentile(b, 97.5))
    report["rmse_ci95"] = ci

    # Wilcoxon on paired squared errors, Holm-corrected
    se = {k: (y.values - p) ** 2 for k, p in oof0.items()}
    raw = {f"XGBoost vs {k}": wilcoxon(se["XGBoost"], se[k]).pvalue for k in se if k != "XGBoost"}
    report["wilcoxon_holm"] = {k: {"p_raw": raw[k], "p_holm": v} for k, v in holm(raw).items()}

    # Humid-only (Am) performance from the same OOF predictions
    am = (df["climate_zone"] == "Am").values
    report["humid_only_Am"] = {k: metrics(y[am], p[am]) for k, p in oof0.items() if k in ("XGBoost", "GBM (Friedman)", "Mean predictor")}

    # Technology ranking (what a storage manager needs)
    t = pd.DataFrame({"tech": df["Storage technology"].str.strip(), "obs": y, "pred": oof0["XGBoost"]})
    tm = t.groupby("tech")[["obs", "pred"]].mean()
    report["tech_ranking_spearman"] = spearmanr(tm.obs, tm.pred).correlation
    report["tech_means"] = tm.round(3).to_dict()
    hm = t.assign(h=X["hermetic"]).groupby("h")[["obs", "pred"]].mean()
    report["hermetic_vs_open"] = hm.round(3).to_dict()

    # 3. Secondary: leave-one-site-out (new-location test, 3 folds)
    lo = oof_predict(X, y, list(LeaveOneGroupOut().split(X, y, site)), 0, log_target)
    report["leave_one_site_out"] = {k: metrics(y, p) for k, p in lo.items()}
    report["loso_per_site_XGB"] = {df.loc[site == s, "site"].iloc[0]: metrics(y[site == s], lo["XGBoost"][(site == s).values])
                                   for s in sorted(site.unique())}

    # 4. SHAP on the final XGBoost fitted to all data
    final = models(0, log_target)["XGBoost"].fit(X, np.log1p(y) if log_target else y)
    sv = shap.TreeExplainer(final).shap_values(X)
    imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
    report["shap_top15"] = imp.head(15).round(4).to_dict()
    final.save_model(f"{outdir}/xgb_final.json")

    # Plots
    plt.figure(figsize=(7, 5)); imp.head(12)[::-1].plot.barh(color="#2f6f4f")
    plt.xlabel("mean |SHAP| (log1p weight loss)" if log_target else "mean |SHAP| (% weight loss)")
    plt.title("XGBoost feature importance (SHAP)"); plt.tight_layout(); plt.savefig(f"{outdir}/shap_importance.png", dpi=200); plt.close()
    plt.figure(figsize=(5.5, 5.5)); c = np.where(am, "#1f77b4", "#d62728")
    plt.scatter(y, oof0["XGBoost"], c=c, s=22, alpha=.75); mx = max(y.max(), oof0["XGBoost"].max()) * 1.05
    plt.plot([0, mx], [0, mx], "k--", lw=1); plt.xlabel("Observed weight loss (%)"); plt.ylabel("Predicted (%)")
    plt.title("XGBoost out-of-fold predictions\nblue = humid (Am), red = sub-humid (Aw)"); plt.tight_layout()
    plt.savefig(f"{outdir}/pred_vs_obs.png", dpi=200); plt.close()

    pd.DataFrame({"row_id": df["row_id"], "site": df["site"], "tech": t.tech, "observed": y,
                  **{f"pred_{k}": v for k, v in oof0.items()}}).to_csv(f"{outdir}/oof_predictions.csv", index=False)
    with open(f"{outdir}/results.json", "w") as f: json.dump(report, f, indent=2, default=float)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default=".")
    a = ap.parse_args(); r = main(a.data, a.out)
    print(json.dumps({k: r[k] for k in ["noise_floor_RMSE", "max_attainable_R2", "log_target_used", "target_transform_RMSE"]}, indent=1, default=float))
    for k, v in r["grouped_cv"].items(): print(f"{k:16s} " + "  ".join(f"{m}={a_:.3f}±{b:.3f}" for m, (a_, b) in v.items()))
    print("CI", r["rmse_ci95"]); print("Wilcoxon", r["wilcoxon_holm"])
    print("Humid", r["humid_only_Am"]); print("Tech rho", r["tech_ranking_spearman"], r["hermetic_vs_open"])
    print("LOSO", r["leave_one_site_out"]); print("LOSO per site", r["loso_per_site_XGB"]); print("SHAP", r["shap_top15"])