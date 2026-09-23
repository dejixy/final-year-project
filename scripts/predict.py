"""
Maize storage weight-loss predictor (CLI).

  python predict.py                     interactive prompts
  python predict.py --tech PP --days 180 --tmin 20 --tmax 31 --rhmin 56 --rhmax 94 --gmci 11.5
  python predict.py --compare --days 180 --tmin 20 --tmax 31 --rhmin 56 --rhmax 94 --gmci 11.5
  python predict.py --batch lots.csv    one row per lot, same column names as the flags

Model: XGBoost trained on 152 CIMMYT lowland-tropical storage lots (3 sites, Mexico).
Typical error is about 2.3 percentage points, so treat output as a risk indicator, not an exact figure.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "xgb_final.json")
DEFAULTS = os.path.join(ROOT, "models", "predictor_defaults.json")

TECHS = {"HBT": "GrainPro SuperGrainbag Farm (hermetic)", "HBZ": "GrainPro SuperGrainbag Premium RZ (hermetic)",
         "HMS": "Hermetic metal silo", "PBO": "Plastic bottle (sealed)", "PBA": "Plastic barrel (sealed)",
         "SPB": "Silage plastic bag (sealed)", "PP": "Polypropylene bag (untreated)",
         "PP_AP": "PP bag + aluminium phosphide", "PP_ML": "PP bag + micronised lime",
         "PP_SL": "PP bag + standard lime"}
HERMETIC = {"HBT", "HBZ", "HMS", "PBO", "PBA", "SPB"}
BANDS = [(1.0, "LOW", "Routine monitoring is enough."),
         (3.0, "MODERATE", "Inspect monthly; consider drying further or selling earlier."),
         (6.0, "HIGH", "Act soon: dry, treat, re-bag into sealed storage, or sell."),
         (float("inf"), "SEVERE", "Intervene now; losses at this level are rarely recoverable.")]


def emc_wb(T, RH):
    """Modified Henderson EMC for yellow dent corn (ASABE D245.6), wet basis %."""
    K, N, C = 8.6541e-5, 1.8634, 49.810
    rh = np.clip(RH / 100.0, 0.01, 0.99)
    mdb = (-np.log(1 - rh) / (K * (T + C))) ** (1 / N)
    return 100 * mdb / (100 + mdb)


def make_row(d, cfg):
    """Build one feature row in the exact order the model was trained on."""
    f = dict(cfg["fallbacks"])          # site-level values the user rarely knows
    f.update({k: v for k, v in d.items() if v is not None})
    tech = str(f["tech"]).upper()
    if tech not in TECHS:
        sys.exit(f"Unknown storage technology '{tech}'. Options: {', '.join(TECHS)}")
    t_mean, rh_mean, days = (f["tmin"] + f["tmax"]) / 2, (f["rhmin"] + f["rhmax"]) / 2, f["days"]
    herm = int(tech in HERMETIC)
    v = {"Elevation (m asl)": f["elevation"], "Tmin (°C)": f["tmin"], "Tmax (°C)": f["tmax"],
         "Hrmin  (%)": f["rhmin"], "Hrmax  (%)": f["rhmax"], "Storage time (days)": days,
         "GMCI  (%)": f["gmci"], "GTI  (°C)": f["gti"], "Grain Impurities (%)": f["impurities"],
         "InsI  (%)": f["insi"], "FunI  (%)": f["funi"], "TdamI  (%)": f["tdami"],
         "LMWI": f["lmwi"], "LLGBI": f["llgbi"], "native_variety": int(f["variety"] == "native"),
         "hermetic": herm, "T_mean": t_mean, "RH_mean": rh_mean, "EMC_wb": emc_wb(t_mean, rh_mean),
         "EMC_gap": f["gmci"] - emc_wb(t_mean, rh_mean),
         "degree_days_15": max(t_mean - 15, 0) * days, "humid_days_70": max(rh_mean - 70, 0) * days,
         "exposure_open": days * (1 - herm)}
    for t in TECHS:
        v[f"tech_{t}"] = int(t == tech)
    return pd.DataFrame([[v[c] for c in cfg["features"]]], columns=cfg["features"])


def band(p):
    for hi, label, advice in BANDS:
        if p < hi:
            return label, advice


def ask(prompt, cast=float, default=None):
    s = input(f"{prompt}{f' [{default}]' if default is not None else ''}: ").strip()
    if not s and default is not None:
        return default
    try:
        return cast(s)
    except ValueError:
        print("  Not a valid value, try again.")
        return ask(prompt, cast, default)


def interactive():
    print("\nMaize storage weight-loss predictor\n" + "-" * 38)
    print("Storage options:")
    for k, v in TECHS.items():
        print(f"  {k:6s} {v}")
    d = {"tech": ask("\nStorage technology code", str, "PP"),
         "days": ask("Storage duration (days)", float, 180),
         "tmin": ask("Average minimum temperature (°C)", float, 21),
         "tmax": ask("Average maximum temperature (°C)", float, 31),
         "rhmin": ask("Average minimum relative humidity (%)", float, 56),
         "rhmax": ask("Average maximum relative humidity (%)", float, 94),
         "gmci": ask("Grain moisture content at intake (%)", float, 12.0)}
    more = input("Enter optional grain-condition details? (y/N): ").strip().lower().startswith("y")
    if more:
        d.update({"impurities": ask("Grain impurities (%)", float, 0.5),
                  "tdami": ask("Total damaged grain at intake (%)", float, 5.0),
                  "insi": ask("Insect-damaged grain at intake (%)", float, 1.0),
                  "variety": ask("Variety (hybrid/native)", str, "hybrid")})
    return d


def report(d, pred, cfg):
    label, advice = band(pred)
    t_mean, rh_mean = (d["tmin"] + d["tmax"]) / 2, (d["rhmin"] + d["rhmax"]) / 2
    gap = d["gmci"] - emc_wb(t_mean, rh_mean)
    print("\n" + "=" * 52)
    print(f"  {TECHS[str(d['tech']).upper()]}")
    print(f"  {d['days']:.0f} days · {t_mean:.1f} °C mean · {rh_mean:.0f} % RH mean · {d['gmci']:.1f} % moisture")
    print("-" * 52)
    print(f"  PREDICTED WEIGHT LOSS   {pred:5.2f} %   [{label} RISK]")
    print(f"  Expected range          {max(pred - cfg['rmse'], 0):5.2f} – {pred + cfg['rmse']:.2f} %   (±1 RMSE)")
    print("-" * 52)
    print(f"  {advice}")
    if gap > 0.5:
        print(f"  Grain is {gap:.1f} pp above its equilibrium moisture — it will lose moisture,")
        print("  but fungal risk is raised while it does. Drying further would help.")
    elif gap < -0.5:
        print(f"  Grain is {abs(gap):.1f} pp below equilibrium — it will absorb moisture from the air.")
        print("  A sealed structure matters here.")
    print("=" * 52)
    print("  Model: XGBoost, 152 CIMMYT lowland-tropical lots (Mexico). Typical error "
          f"±{cfg['rmse']:.1f} pp.\n  A risk indicator, not a measurement. Not validated on Nigerian data.\n")


def main():
    p = argparse.ArgumentParser(description="Predict maize post-harvest weight loss in storage.")
    for f, h in [("tech", "storage technology code"), ("variety", "hybrid or native")]:
        p.add_argument(f"--{f}", help=h)

    for f, h in [("days", "storage duration (days)"), ("tmin", "min temperature °C"), ("tmax", "max temperature °C"),
                 ("rhmin", "min relative humidity %%"), ("rhmax", "max relative humidity %%"),
                 ("gmci", "intake moisture content %%"), ("impurities", "grain impurities %%"),
                 ("insi", "insect-damaged grain at intake %%"), ("tdami", "total damage at intake %%")]:
        
        p.add_argument(f"--{f}", type=float, help=h)
    p.add_argument("--compare", action="store_true", help="rank every storage technology for these conditions")
    p.add_argument("--batch", help="CSV of lots to score; writes <name>_predictions.csv")
    p.add_argument("--list-tech", action="store_true", help="list storage technology codes and exit")
    a = p.parse_args()

    if a.list_tech:
        for k, v in TECHS.items():
            print(f"{k:6s} {v}")
        return
    if not os.path.exists(MODEL):
        sys.exit("xgb_final.json not found — run build_xgboost.py first.")
    cfg = json.load(open(DEFAULTS))
    model = XGBRegressor(); model.load_model(MODEL)
    predict = lambda d: float(np.clip(model.predict(make_row(d, cfg))[0], 0, None))

    if a.batch:
        df = pd.read_csv(a.batch)
        df["predicted_weight_loss_pct"] = [predict(r._asdict()) for r in df.itertuples(index=False)]
        df["risk"] = [band(v)[0] for v in df.predicted_weight_loss_pct]
        out = a.batch.rsplit(".", 1)[0] + "_predictions.csv"
        df.to_csv(out, index=False)
        print(f"Wrote {out} ({len(df)} lots)")
        return

    given = {k: getattr(a, k) for k in ("tech", "days", "tmin", "tmax", "rhmin", "rhmax", "gmci",
                                        "impurities", "insi", "tdami", "variety")}
    need = ["days", "tmin", "tmax", "rhmin", "rhmax", "gmci"]
    d = given if all(given[k] is not None for k in need) else {**given, **interactive()}
    d = {k: v for k, v in d.items() if v is not None}

    if a.compare:
        rows = sorted(((t, predict({**d, "tech": t})) for t in TECHS), key=lambda x: x[1])
        print(f"\nRanked storage options — {d['days']:.0f} days at {d['gmci']:.1f} % intake moisture\n" + "-" * 62)
        for t, v in rows:
            print(f"  {v:5.2f} %  {band(v)[0]:8s} {t:6s} {TECHS[t]}")
        print("-" * 62 + "\n  Ranking is more reliable than the absolute numbers (Spearman 0.95).\n")
    else:
        d.setdefault("tech", "PP")
        report(d, predict(d), cfg)


if __name__ == "__main__":
    main()