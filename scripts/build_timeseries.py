"""
Time-aware component: deterioration curves for stored maize.

Why not ARIMA/LSTM: the trials give 2-3 sampling points per storage unit, not a weekly
panel. Sequence models cannot be identified from that. Instead loss is modelled as a
function of storage time per technology class, which yields deterioration rates and safe
storage times - the operational output Section 3.5.3.6 specifies.

  python build_timeseries.py --data ../cimmyt_lowland_tropical_merged.csv --out .
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings("ignore")

TARGET = "Weight_Loss  (%)"
DAYS = "Storage time (days)"
HERMETIC = {"HBT", "HBZ", "HMS", "PBO", "PBA", "SPB"}
TREATED = {"PP_AP", "PP_ML", "PP_SL"}
THRESHOLDS = [1.0, 3.0, 5.0]

# Candidate shapes. t in days, scaled by 100 so parameters stay well conditioned.
def linear(t, a):        return a * (t / 100)
def power(t, a, b):      return a * (t / 100) ** b
def expo(t, a, k):       return a * (np.exp(k * t / 100) - 1)
MODELS = {"linear": (linear, [1.0], ([0], [np.inf])),
          "power":  (power,  [1.0, 1.5], ([0, 0.2], [np.inf, 5])),
          "expo":   (expo,   [1.0, 0.5], ([0, 0.01], [np.inf, 5]))}


def klass(tech):
    return "hermetic" if tech in HERMETIC else ("pp_treated" if tech in TREATED else "pp_open")


def fit(name, t, y):
    f, p0, bnd = MODELS[name]
    try:
        p, _ = curve_fit(f, t, y, p0=p0, bounds=bnd, maxfev=20000)
        resid = y - f(t, *p)
        k = len(p)
        rss = float(np.sum(resid ** 2))
        aic = len(y) * np.log(max(rss, 1e-9) / len(y)) + 2 * k
        return {"params": [float(v) for v in p], "rss": rss, "aic": float(aic)}
    except Exception:
        return None


def predict(name, params, t):
    return np.clip(MODELS[name][0](np.asarray(t, float), *params), 0, None)


def safe_time(name, params, thr, tmax=400):
    """Days until the fitted curve reaches a loss threshold."""
    t = np.arange(1, tmax + 1)
    v = predict(name, params, t)
    hit = np.where(v >= thr)[0]
    return int(t[hit[0]]) if len(hit) else None


def metrics(y, p):
    return {"RMSE": mean_squared_error(y, p) ** 0.5, "MAE": mean_absolute_error(y, p),
            "R2": r2_score(y, p), "Bias(obs-pred)": float(np.mean(y - p))}


def main(path, out):
    d = pd.read_csv(path)
    d["tech"] = d["Storage technology"].str.strip()
    d["class"] = d["tech"].map(klass)
    d["unit"] = d["site"] + " " + d["Year"].astype(str) + " " + d["tech"]
    rep = {"n_rows": len(d), "n_units": int(d.unit.nunique()),
           "units_multi_timepoint": int((d.groupby("unit")[DAYS].nunique() >= 2).sum()),
           "rows_in_multi_units": int(d[d.unit.isin(d.groupby("unit")[DAYS].nunique().pipe(lambda s: s[s >= 2]).index)].shape[0]),
           "why_not_arima": "2-3 sampling points per unit; ARIMA/ARIMAX/LSTM are not identifiable on series of that length."}

    # ---- 1. Curve selection per technology class (fitted on all rows of that class)
    rep["class_fits"] = {}
    chosen = {}
    for c, g in d.groupby("class"):
        fits = {n: fit(n, g[DAYS].values, g[TARGET].values) for n in MODELS}
        fits = {n: v for n, v in fits.items() if v}
        best = min(fits, key=lambda n: fits[n]["aic"])
        chosen[c] = (best, fits[best]["params"])
        rep["class_fits"][c] = {"n": len(g), "aic": {n: round(v["aic"], 1) for n, v in fits.items()},
                                "selected": best, "params": [round(v, 4) for v in fits[best]["params"]]}

    # ---- 2. Per-technology curves, using the class shape, refitted for scale
    rep["tech_curves"] = {}
    tech_par = {}
    for t_, g in d.groupby("tech"):
        shape = chosen[klass(t_)][0]
        if g[DAYS].nunique() < 2:                      # single duration: scale-only fit
            f = fit("linear", g[DAYS].values, g[TARGET].values)
            shape, par = "linear", f["params"]
        else:
            f = fit(shape, g[DAYS].values, g[TARGET].values) or fit("linear", g[DAYS].values, g[TARGET].values)
            shape = shape if f else "linear"; par = f["params"]
        tech_par[t_] = (shape, par)
        rep["tech_curves"][t_] = {"n": len(g), "durations": int(g[DAYS].nunique()), "shape": shape,
                                  "params": [round(v, 4) for v in par],
                                  "loss_at": {f"{dd}d": round(float(predict(shape, par, [dd])[0]), 2)
                                              for dd in (30, 90, 180, 270)},
                                  "days_to": {f"{thr}%": safe_time(shape, par, thr) for thr in THRESHOLDS}}

    # ---- 3. Leave-one-unit-out validation, class curves vs naive benchmarks
    rows, preds = [], {"curve": [], "mean": [], "persistence": [], "tech_mean": []}
    for u in sorted(d.unit.unique()):
        te, tr = d[d.unit == u], d[d.unit != u]
        c = te["class"].iloc[0]
        trc = tr[tr["class"] == c]
        if len(trc) < 4:
            continue
        shape = chosen[c][0]
        f = fit(shape, trc[DAYS].values, trc[TARGET].values) or fit("linear", trc[DAYS].values, trc[TARGET].values)
        sh = shape if f else "linear"
        preds["curve"] += list(predict(sh, f["params"], te[DAYS].values))
        preds["mean"] += [tr[TARGET].mean()] * len(te)
        preds["tech_mean"] += [tr[tr.tech == te.tech.iloc[0]][TARGET].mean() if (tr.tech == te.tech.iloc[0]).any()
                               else tr[TARGET].mean()] * len(te)
        # persistence: carry the earliest observed value of this unit forward
        first = te.loc[te[DAYS] == te[DAYS].min(), TARGET].mean()
        preds["persistence"] += [first] * len(te)
        rows.append(te)
    obs = pd.concat(rows)[TARGET].values
    rep["leave_one_unit_out"] = {k: metrics(obs, np.array(v)) for k, v in preds.items()}
    rep["n_eval_rows"] = len(obs)

    # ---- 4. Same task for XGBoost, aligned information set (duration + conditions, no unit)
    try:
        import sys, importlib.util
        _here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location("b", os.path.join(_here, "build_xgboost.py"))
        b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
        X, y = b.build_features(d), d[TARGET]
        from xgboost import XGBRegressor
        xp = np.zeros(len(d))
        for u in sorted(d.unit.unique()):
            te = (d.unit == u).values
            m = XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=3,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="exact",
                             n_jobs=1, random_state=0).fit(X[~te], y[~te])
            xp[te] = np.clip(m.predict(X[te]), 0, None)
        keep = d.unit.isin(pd.concat(rows).unit.unique()).values
        rep["leave_one_unit_out"]["xgboost"] = metrics(y[keep].values, xp[keep])
    except Exception as e:
        rep["xgboost_comparison_error"] = str(e)

    # ---- 5. Plot: observed points and fitted curves by class
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    for i, (c, ttl) in enumerate([("hermetic", "Hermetic / sealed"), ("pp_treated", "PP + treatment"),
                                  ("pp_open", "PP untreated")]):
        g = d[d["class"] == c]; sh, par = chosen[c]
        ax[i].scatter(g[DAYS], g[TARGET], s=16, alpha=.55, color="#2f6f4f" if c == "hermetic" else
                      ("#b8860b" if c == "pp_treated" else "#a33"))
        tt = np.arange(0, 275)
        ax[i].plot(tt, predict(sh, par, tt), color="#222", lw=1.8)
        for thr in THRESHOLDS:
            st = safe_time(sh, par, thr)
            if st and st <= 270:
                ax[i].axvline(st, ls=":", lw=1, color="#777")
                ax[i].text(st + 3, ax[i].get_ylim()[1] * .92, f"{thr:g}%", fontsize=8, color="#555")
        ax[i].set_title(f"{ttl} · {sh}", fontsize=10); ax[i].set_xlabel("days in store")
        if i == 0: ax[i].set_ylabel("weight loss (%)")
    plt.tight_layout(); plt.savefig(f"{out}/deterioration_curves.png", dpi=200); plt.close()

    json.dump(rep, open(f"{out}/timeseries_results.json", "w"), indent=2, default=float)
    json.dump({"class_curves": {c: {"shape": s, "params": p} for c, (s, p) in chosen.items()},
               "tech_curves": {t: {"shape": s, "params": p} for t, (s, p) in tech_par.items()}},
              open(f"{out}/curve_params.json", "w"), indent=2)
    return rep


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--data", required=True); a.add_argument("--out", default=".")
    ar = a.parse_args(); r = main(ar.data, ar.out)
    print(json.dumps({k: r[k] for k in ("n_units", "units_multi_timepoint", "rows_in_multi_units",
                                         "n_eval_rows", "class_fits")}, indent=1, default=float))
    print("\nLeave-one-unit-out:")
    for k, v in r["leave_one_unit_out"].items():
        print(f"  {k:12s} " + "  ".join(f"{m}={x:.3f}" for m, x in v.items()))
    print("\nSafe storage times (days to reach loss threshold):")
    for t_, v in r["tech_curves"].items():
        print(f"  {t_:6s} {v['shape']:7s} n={v['n']:3d} durations={v['durations']}  "
              f"loss@180d={v['loss_at']['180d']:5.2f}  to1%={v['days_to']['1.0%']}  to5%={v['days_to']['5.0%']}")