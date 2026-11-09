"""
Project 2 - Digital Advertising Budget Optimizer
=================================================
Industry problem
----------------
A company advertises on Facebook, Instagram, Google, TikTok and YouTube.
Management asks: "Where should we spend the next Rs. 1,000,000?"

This script:
  1. Generates a realistic ad-campaign dataset (12 columns, 10,000 base records).
  2. Computes the standard marketing metrics: CTR, CPC, CPA, conversion rate,
     ROAS, revenue, and per-campaign performance.
  3. Runs a Big-Data scaling simulation (10k -> 100k -> 1M -> 10M records) and
     compares three compute styles: Pandas vs MapReduce vs Spark, cross-checking
     that all three produce the SAME aggregation (correctness, not just speed).
  4. Builds a Marketing Budget Recommendation Engine that splits the next
     Rs. 1,000,000 across platforms based on simulated historical performance.
  5. Ends with a REQUIREMENTS CHECK (PASS/FAIL + non-zero exit on failure).

Run:  python project2_budget_optimizer.py

Honesty notes: the data is SIMULATED (clearly labelled), so this demonstrates
the workflow, not a real market claim. The recommendation assumes past
performance persists and ignores diminishing returns / audience saturation;
it should be validated with controlled A/B tests before real spend.
"""

import os
import sys
import glob
import time
import shutil
import tempfile
import functools
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs_project2")
CSV = os.path.join(DATA_DIR, "project2_ad_campaigns.csv")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

PLATFORMS = ["Facebook", "Instagram", "Google", "TikTok", "YouTube"]
BUDGET = 1_000_000          # Rs. 1 million to allocate
BASE_RECORDS = 10_000       # teaching baseline
RNG = np.random.default_rng(42)

# Spark on Windows/Hadoop breaks on paths containing spaces (and this project
# lives under "Big Data Analytics"), so Spark reads from a space-free staging dir.
SPARK_STAGE = os.path.join(tempfile.gettempdir(), "spark_stage_proj2")

RESULTS = {}


def banner(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def savefig(name):
    plt.tight_layout()
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=110)
    plt.close()
    print(f"   saved chart -> outputs_project2/{name}")


# ==========================================================================
# 1 · DATA GENERATION
# ==========================================================================
# Each platform is given DIFFERENT "true" performance so the recommendation
# engine has a real, defensible signal (not random noise).
PLATFORM_PROFILE = {
    #            base_ctr  cvr    cpc_rs  rev_per_conv  impressions_scale
    "Facebook":  (0.018,  0.055,  12.0,   900,   1.0),
    "Instagram": (0.022,  0.050,  14.0,   950,   1.1),
    "Google":    (0.035,  0.075,  22.0,  1200,   0.9),   # high intent, pricey
    "TikTok":    (0.028,  0.030,   8.0,   700,   1.3),   # cheap, lower convert
    "YouTube":   (0.015,  0.040,  10.0,   850,   1.0),
}
CAMPAIGNS = ["Awareness", "Retargeting", "Conversion", "Seasonal", "Launch"]
GENDERS = ["Male", "Female", "Other"]
CITIES = ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad",
          "Multan", "Peshawar", "Quetta"]
DEVICES = ["Mobile", "Desktop", "Tablet"]


def generate_data(n, rng=RNG):
    """Vectorised generation of n ad-campaign records (12 columns)."""
    plat_idx = rng.integers(0, len(PLATFORMS), n)
    platform = np.array(PLATFORMS)[plat_idx]

    # Pull per-row platform parameters
    base_ctr = np.array([PLATFORM_PROFILE[p][0] for p in PLATFORMS])[plat_idx]
    cvr = np.array([PLATFORM_PROFILE[p][1] for p in PLATFORMS])[plat_idx]
    cpc = np.array([PLATFORM_PROFILE[p][2] for p in PLATFORMS])[plat_idx]
    rpc = np.array([PLATFORM_PROFILE[p][3] for p in PLATFORMS])[plat_idx]
    imp_scale = np.array([PLATFORM_PROFILE[p][4] for p in PLATFORMS])[plat_idx]

    # Impressions per record (log-normal spread), then clicks/conv/spend/rev
    impressions = (rng.lognormal(mean=8.0, sigma=0.6, size=n) * imp_scale).astype(int) + 100
    ctr = np.clip(base_ctr * rng.normal(1.0, 0.20, n), 0.001, 0.5)
    clicks = np.random.default_rng().binomial(impressions, ctr)
    conv_rate = np.clip(cvr * rng.normal(1.0, 0.25, n), 0.0, 0.6)
    conversions = np.random.default_rng().binomial(np.maximum(clicks, 0), conv_rate)
    spend = np.round(clicks * cpc * rng.normal(1.0, 0.15, n), 2)
    revenue = np.round(conversions * rpc * rng.normal(1.0, 0.20, n), 2)

    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")

    df = pd.DataFrame({
        "date": dates,
        "campaign": np.array(CAMPAIGNS)[rng.integers(0, len(CAMPAIGNS), n)],
        "platform": platform,
        "impressions": impressions,
        "clicks": clicks,
        "spend": spend,
        "conversions": conversions,
        "revenue": revenue,
        "age": rng.integers(18, 65, n),
        "gender": np.array(GENDERS)[rng.integers(0, len(GENDERS), n)],
        "city": np.array(CITIES)[rng.integers(0, len(CITIES), n)],
        "device": np.array(DEVICES)[rng.integers(0, len(DEVICES), n)],
    })
    return df


# ==========================================================================
# 2 · MARKETING METRICS
# ==========================================================================
def add_metrics(df):
    """Row-level marketing KPIs. Guard divide-by-zero (no clicks/conv/spend)."""
    out = df.copy()
    out["CTR"] = np.where(out["impressions"] > 0, out["clicks"] / out["impressions"], np.nan)
    out["CPC"] = np.where(out["clicks"] > 0, out["spend"] / out["clicks"], np.nan)
    out["CPA"] = np.where(out["conversions"] > 0, out["spend"] / out["conversions"], np.nan)
    out["conversion_rate"] = np.where(out["clicks"] > 0, out["conversions"] / out["clicks"], np.nan)
    out["ROAS"] = np.where(out["spend"] > 0, out["revenue"] / out["spend"], np.nan)
    return out


def campaign_performance(df):
    """Per campaign x platform performance summary."""
    g = (df.groupby(["campaign", "platform"])
           .agg(impressions=("impressions", "sum"),
                clicks=("clicks", "sum"),
                spend=("spend", "sum"),
                conversions=("conversions", "sum"),
                revenue=("revenue", "sum"))
           .reset_index())
    g["CTR"] = g["clicks"] / g["impressions"]
    g["CPA"] = np.where(g["conversions"] > 0, g["spend"] / g["conversions"], np.nan)
    g["ROAS"] = np.where(g["spend"] > 0, g["revenue"] / g["spend"], np.nan)
    return g


# ==========================================================================
# 3 · BIG DATA SIMULATION — Pandas vs MapReduce vs Spark
# ==========================================================================
# The shared task: aggregate spend, revenue, conversions by platform, then ROAS.

def agg_pandas(df):
    g = df.groupby("platform")[["spend", "revenue", "conversions"]].sum()
    g["ROAS"] = g["revenue"] / g["spend"]
    return g.sort_index()


def _mapper(record):
    # map: emit (platform, (spend, revenue, conversions))
    plat, spend, rev, conv = record
    return (plat, (spend, rev, conv))


def _reducer(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def agg_mapreduce(df):
    """Genuine map -> shuffle(group) -> reduce in pure Python."""
    records = zip(df["platform"].to_numpy(), df["spend"].to_numpy(),
                  df["revenue"].to_numpy(), df["conversions"].to_numpy())
    # MAP
    mapped = map(_mapper, records)
    # SHUFFLE (group by key)
    shuffled = {}
    for key, val in mapped:
        shuffled.setdefault(key, []).append(val)
    # REDUCE
    reduced = {k: functools.reduce(_reducer, vals) for k, vals in shuffled.items()}
    out = pd.DataFrame(
        [(k, v[0], v[1], v[2]) for k, v in reduced.items()],
        columns=["platform", "spend", "revenue", "conversions"]
    ).set_index("platform")
    out["ROAS"] = out["revenue"] / out["spend"]
    return out.sort_index()


# ---- Spark (real if available, else honest multiprocessing stand-in) ------
SPARK = None
SPARK_MODE = "unavailable"


def _find_spark_jdk():
    """Spark 4.x needs Java 17 or 21. Find one; set JAVA_HOME so Spark uses it
    (this machine's PATH 'java' is Java 27, which Spark does not support)."""
    # An already-correct JAVA_HOME wins.
    jh = os.environ.get("JAVA_HOME", "")
    if jh and any(f"jdk-{v}" in jh for v in ("17", "21")) and os.path.isdir(jh):
        return jh
    patterns = [
        r"C:\Program Files\Eclipse Adoptium\jdk-21*",
        r"C:\Program Files\Eclipse Adoptium\jdk-17*",
        r"C:\Program Files\Java\jdk-21*",
        r"C:\Program Files\Java\jdk-17*",
        r"C:\Program Files\Microsoft\jdk-21*",
        r"C:\Program Files\Microsoft\jdk-17*",
    ]
    for pat in patterns:
        hits = [h for h in glob.glob(pat) if os.path.isdir(h)]
        if hits:
            return sorted(hits)[-1]
    return None


def _init_spark():
    """Start a real local Spark if a compatible JDK + pyspark exist, else mark
    the multiprocessing stand-in. Validation uses a JVM-only job (spark.range)
    because PySpark 4.3's Python *worker* crashes on Python 3.13."""
    global SPARK, SPARK_MODE
    try:
        jdk = _find_spark_jdk()
        if jdk:
            os.environ["JAVA_HOME"] = jdk
        from pyspark.sql import SparkSession
        os.makedirs(SPARK_STAGE, exist_ok=True)
        warehouse = os.path.join(SPARK_STAGE, "warehouse").replace("\\", "/")
        SPARK = (SparkSession.builder
                 .appName("BudgetOptimizer")
                 .master("local[*]")
                 .config("spark.ui.enabled", "false")
                 .config("spark.sql.shuffle.partitions", "8")
                 .config("spark.sql.warehouse.dir", f"file:///{warehouse}")
                 .getOrCreate())
        SPARK.sparkContext.setLogLevel("ERROR")
        # JVM-only smoke test (no Python worker, no file IO)
        import pyspark.sql.functions as F
        SPARK.range(0, 1000).groupBy((F.col("id") % 5)).count().collect()
        SPARK_MODE = "pyspark"
        globals()["_SPARK_JDK"] = jdk
    except Exception as e:
        SPARK = None
        SPARK_MODE = "multiprocessing"
        globals()["_SPARK_ERR"] = str(e).splitlines()[0][:140]


def _chunk_partial(args):
    plats, spend, rev, conv = args
    acc = {}
    for p, s, r, c in zip(plats, spend, rev, conv):
        if p in acc:
            acc[p][0] += s; acc[p][1] += r; acc[p][2] += c
        else:
            acc[p] = [s, r, c]
    return acc


def agg_spark(df, staged_uri=None):
    """Real Spark groupBy if pyspark works; else a parallel multiprocessing
    map/reduce that mimics distributed partition-aggregate-combine.

    Real Spark reads the data from a space-free CSV (staged_uri) with Spark's
    JVM-native reader and an explicit schema — this avoids both the Windows
    path-with-spaces bug and the PySpark-4.3/Python-3.13 worker crash. collect()
    (not toPandas) returns the tiny 5-row result without spawning Python workers."""
    if SPARK_MODE == "pyspark" and staged_uri is not None:
        from pyspark.sql.types import (StructType, StructField, StringType,
                                        DoubleType, LongType)
        import pyspark.sql.functions as F
        schema = StructType([
            StructField("platform", StringType()),
            StructField("spend", DoubleType()),
            StructField("revenue", DoubleType()),
            StructField("conversions", LongType()),
        ])
        sdf = SPARK.read.option("header", True).schema(schema).csv(staged_uri)
        rows = (sdf.groupBy("platform")
                   .agg(F.sum("spend").alias("spend"),
                        F.sum("revenue").alias("revenue"),
                        F.sum("conversions").alias("conversions"))
                   .collect())
        res = pd.DataFrame([r.asDict() for r in rows]).set_index("platform")
        res["ROAS"] = res["revenue"] / res["spend"]
        return res.sort_index()

    # ---- multiprocessing stand-in (map across partitions, reduce/combine) ----
    n = len(df)
    k = min(mp.cpu_count(), 4)
    idx = np.array_split(np.arange(n), k)
    plats = df["platform"].to_numpy(); sp = df["spend"].to_numpy()
    rv = df["revenue"].to_numpy(); cv = df["conversions"].to_numpy()
    parts = [(plats[i], sp[i], rv[i], cv[i]) for i in idx]
    try:
        with mp.Pool(k) as pool:
            partials = pool.map(_chunk_partial, parts)
    except Exception:
        partials = [_chunk_partial(p) for p in parts]  # fallback: serial
    combined = {}
    for acc in partials:
        for p, (s, r, c) in acc.items():
            if p in combined:
                combined[p][0] += s; combined[p][1] += r; combined[p][2] += c
            else:
                combined[p] = [s, r, c]
    out = pd.DataFrame([(p, v[0], v[1], v[2]) for p, v in combined.items()],
                       columns=["platform", "spend", "revenue", "conversions"]
                       ).set_index("platform")
    out["ROAS"] = out["revenue"] / out["spend"]
    return out.sort_index()


def results_match(a, b, tol=1.0):
    """Compare two aggregation frames (spend/revenue/conversions) within tol."""
    a2 = a[["spend", "revenue", "conversions"]].sort_index()
    b2 = b[["spend", "revenue", "conversions"]].sort_index()
    if list(a2.index) != list(b2.index):
        return False
    return bool(np.allclose(a2.to_numpy(), b2.to_numpy(), atol=tol, rtol=1e-6))


def scaling_simulation(base_df):
    banner("BIG DATA SIMULATION · Pandas vs MapReduce vs Spark")
    print(f"Spark mode: {SPARK_MODE}"
          + (f"  (real Spark on JDK: {globals().get('_SPARK_JDK','?')})"
             if SPARK_MODE == "pyspark"
             else f"  (pyspark unavailable: {globals().get('_SPARK_ERR','n/a')})"))
    print("Task: aggregate spend/revenue/conversions by platform, then ROAS.\n")

    scales = [10_000, 100_000, 1_000_000, 10_000_000]
    rows = []
    all_match = True
    for n in scales:
        # Build n rows by tiling the base sample (a scaling experiment, not new data)
        reps = int(np.ceil(n / len(base_df)))
        big = pd.concat([base_df] * reps, ignore_index=True).iloc[:n]

        t0 = time.perf_counter(); r_pd = agg_pandas(big); t_pd = time.perf_counter() - t0
        t0 = time.perf_counter(); r_mr = agg_mapreduce(big); t_mr = time.perf_counter() - t0

        # For real Spark, stage the SAME rows to a space-free CSV (this is the
        # "data already in storage" step, not timed), then time Spark's read+agg.
        staged_uri = None
        if SPARK_MODE == "pyspark":
            os.makedirs(SPARK_STAGE, exist_ok=True)
            stage_csv = os.path.join(SPARK_STAGE, f"scale_{n}.csv")
            big[["platform", "spend", "revenue", "conversions"]].to_csv(stage_csv, index=False)
            staged_uri = "file:///" + os.path.abspath(stage_csv).replace("\\", "/")
        t0 = time.perf_counter(); r_sp = agg_spark(big, staged_uri); t_sp = time.perf_counter() - t0
        if staged_uri:
            try:
                os.remove(stage_csv)
            except OSError:
                pass

        match = results_match(r_pd, r_mr) and results_match(r_pd, r_sp)
        all_match = all_match and match
        rows.append({"rows": n, "pandas_s": round(t_pd, 4),
                     "mapreduce_s": round(t_mr, 4), "spark_s": round(t_sp, 4),
                     "results_agree": match})
        print(f"  {n:>10,} rows | pandas {t_pd:7.3f}s | mapreduce {t_mr:7.3f}s | "
              f"spark/{SPARK_MODE[:4]} {t_sp:7.3f}s | agree={match}")
        del big

    scale_df = pd.DataFrame(rows)

    # Chart: compute time vs rows
    ax = plt.gca()
    for col, lab in [("pandas_s", "Pandas"), ("mapreduce_s", "MapReduce"),
                     (f"spark_s", f"Spark ({SPARK_MODE})")]:
        ax.plot(scale_df["rows"], scale_df[col], marker="o", label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Records (log)"); ax.set_ylabel("Time seconds (log)")
    ax.set_title("Aggregation time: Pandas vs MapReduce vs Spark")
    ax.legend()
    savefig("scaling_pandas_mapreduce_spark.png")

    print("\nRead: on this single machine Pandas (vectorised C) is fastest;")
    print("pure-Python MapReduce scales linearly but is slowest; Spark here reads")
    print("the data from disk and aggregates in the JVM — its start-up and IO")
    print("overhead only pays off on a cluster with data too big for one machine's")
    print("memory. All three produce the SAME RESULT via different engines.")

    RESULTS["scale_df"] = scale_df
    RESULTS["scale_max_rows"] = int(scale_df["rows"].max())
    RESULTS["engines_agree"] = bool(all_match)
    RESULTS["spark_mode"] = SPARK_MODE
    return scale_df


# ==========================================================================
# 4 · MARKETING BUDGET RECOMMENDATION ENGINE
# ==========================================================================
def recommend_budget(df, budget=BUDGET):
    banner("MARKETING BUDGET RECOMMENDATION ENGINE")
    perf = (df.groupby("platform")
              .agg(spend=("spend", "sum"), revenue=("revenue", "sum"),
                   conversions=("conversions", "sum"), clicks=("clicks", "sum"),
                   impressions=("impressions", "sum"))
              .reindex(PLATFORMS))
    perf["ROAS"] = perf["revenue"] / perf["spend"]
    perf["CPA"] = perf["spend"] / perf["conversions"]
    perf["CTR"] = perf["clicks"] / perf["impressions"]

    # Allocation score = ROAS (return per rupee). Normalise to % of budget.
    score = perf["ROAS"].clip(lower=0)
    weights = score / score.sum()
    perf["recommended_pct"] = (weights * 100).round(1)
    perf["recommended_rs"] = (weights * budget).round(0)

    print("Historical performance by platform (simulated):")
    print(perf[["spend", "revenue", "ROAS", "CPA", "recommended_pct",
                "recommended_rs"]].round(2).to_string())
    print(f"\n>>> Recommended split of the next Rs. {budget:,.0f}:")
    for p in PLATFORMS:
        print(f"    {p:10} {perf.loc[p,'recommended_pct']:5.1f}%  "
              f"= Rs. {perf.loc[p,'recommended_rs']:>12,.0f}")
    print(f"    {'TOTAL':10} {perf['recommended_pct'].sum():5.1f}%  "
          f"= Rs. {perf['recommended_rs'].sum():>12,.0f}")

    # Chart
    perf["recommended_pct"].plot(kind="bar", color="#2f9e44",
                                 title="Recommended budget allocation (%)")
    plt.ylabel("% of Rs. 1,000,000"); plt.xlabel("Platform")
    savefig("budget_allocation.png")

    print("\nLimitations: data is SIMULATED; the split assumes past ROAS persists,")
    print("ignores diminishing returns / audience saturation / seasonality, and")
    print("should be confirmed with controlled A/B tests before committing spend.")

    RESULTS["budget_perf"] = perf
    RESULTS["budget_pct_sum"] = float(perf["recommended_pct"].sum())
    RESULTS["budget_rs_sum"] = float(perf["recommended_rs"].sum())
    RESULTS["budget_platforms"] = list(perf.index)
    return perf


# ==========================================================================
# 5 · REQUIREMENTS CHECK
# ==========================================================================
def requirements_check(df, metrics_df, camp_perf):
    banner("REQUIREMENTS CHECK")
    required_cols = ["date", "campaign", "platform", "impressions", "clicks",
                     "spend", "conversions", "revenue", "age", "gender",
                     "city", "device"]
    metric_cols = ["CTR", "CPC", "CPA", "conversion_rate", "ROAS"]

    checks = [
        ("Dataset has all 12 required columns",
         all(c in df.columns for c in required_cols)),
        ("Baseline dataset has 10,000 records",
         len(df) == BASE_RECORDS),
        ("Platforms are exactly the 5 named channels",
         sorted(df["platform"].unique()) == sorted(PLATFORMS)),
        ("Metric: CTR computed", "CTR" in metrics_df.columns),
        ("Metric: CPC computed", "CPC" in metrics_df.columns),
        ("Metric: CPA computed", "CPA" in metrics_df.columns),
        ("Metric: conversion rate computed", "conversion_rate" in metrics_df.columns),
        ("Metric: ROAS computed", "ROAS" in metrics_df.columns),
        ("Metric: revenue present", "revenue" in metrics_df.columns),
        ("Metric: campaign performance table built",
         len(camp_perf) > 0 and "ROAS" in camp_perf.columns),
        ("Metrics are finite where denominators > 0",
         metrics_df.loc[metrics_df["impressions"] > 0, "CTR"].notna().all()),
        ("Big Data sim scaled to 10,000,000 records",
         RESULTS.get("scale_max_rows") == 10_000_000),
        ("Three engines compared (Pandas, MapReduce, Spark)",
         {"pandas_s", "mapreduce_s", "spark_s"}.issubset(RESULTS["scale_df"].columns)),
        ("Engines produce IDENTICAL aggregation (correctness)",
         RESULTS.get("engines_agree") is True),
        ("Budget engine covers all 5 platforms",
         RESULTS.get("budget_platforms") == PLATFORMS),
        ("Budget allocation sums to ~100%",
         abs(RESULTS.get("budget_pct_sum", 0) - 100.0) <= 0.2),
        ("Budget allocation sums to ~Rs. 1,000,000",
         abs(RESULTS.get("budget_rs_sum", 0) - BUDGET) <= 2.0),
    ]
    passed = 0
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        passed += bool(ok)

    expected_charts = ["scaling_pandas_mapreduce_spark.png", "budget_allocation.png"]
    missing = [c for c in expected_charts if not os.path.exists(os.path.join(OUT_DIR, c))]
    charts_ok = not missing
    print(f"  [{'PASS' if charts_ok else 'FAIL'}] all {len(expected_charts)} charts saved"
          + ("" if charts_ok else f" (missing {missing})"))

    print("-" * 72)
    print(f"  {passed}/{len(checks)} requirement checks PASSED"
          + ("" if charts_ok else "  (+ charts FAIL)"))
    all_ok = passed == len(checks) and charts_ok
    print("=" * 72)
    print("  RESULT:", "ALL REQUIREMENTS FULFILLED" if all_ok else "SOME REQUIREMENTS FAILED")
    if RESULTS.get("spark_mode") != "pyspark":
        print("  NOTE: real pyspark was not runnable in this environment; the Spark")
        print("        stage used a multiprocessing map/reduce stand-in (clearly")
        print("        labelled). Install a Java-8/11/17 + pyspark combo for true Spark.")
    print("=" * 72)
    return all_ok


# --------------------------------------------------------------------------
def main():
    banner("1 · GENERATE DATASET (simulated, 12 columns)")
    df = generate_data(BASE_RECORDS)
    df.to_csv(CSV, index=False)
    print(f"Generated {len(df):,} records -> data/project2_ad_campaigns.csv")
    print(df.head().to_string())

    banner("2 · MARKETING METRICS")
    metrics_df = add_metrics(df)
    overall = {
        "CTR":  metrics_df["clicks"].sum() / metrics_df["impressions"].sum(),
        "CPC":  metrics_df["spend"].sum() / metrics_df["clicks"].sum(),
        "CPA":  metrics_df["spend"].sum() / metrics_df["conversions"].sum(),
        "conversion_rate": metrics_df["conversions"].sum() / metrics_df["clicks"].sum(),
        "ROAS": metrics_df["revenue"].sum() / metrics_df["spend"].sum(),
        "revenue": metrics_df["revenue"].sum(),
    }
    print("Overall account metrics:")
    for k, v in overall.items():
        print(f"  {k:16}: {v:,.4f}" if k != "revenue" else f"  {k:16}: Rs. {v:,.0f}")
    camp_perf = campaign_performance(df)
    print(f"\nCampaign performance table: {len(camp_perf)} campaign x platform rows")
    print(camp_perf.head().round(3).to_string(index=False))

    _init_spark()
    scaling_simulation(df)
    recommend_budget(df)
    ok = requirements_check(df, metrics_df, camp_perf)

    if SPARK is not None:
        try:
            SPARK.stop()
        except Exception:
            pass
    try:
        shutil.rmtree(SPARK_STAGE, ignore_errors=True)
    except Exception:
        pass
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
