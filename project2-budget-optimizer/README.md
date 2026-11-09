# Project 2 — Digital Advertising Budget Optimizer

Worked solution for the Project 2 brief. Pure Python, self-testing.

## Industry problem
A company advertises on **Facebook, Instagram, Google, TikTok, YouTube**.
Management asks: *"Where should we spend the next Rs. 1,000,000?"*

## Files
- `project2_budget_optimizer.py` — the whole project (run it).
- `data/project2_ad_campaigns.csv` — generated 10,000-record dataset (12 columns).
- `outputs_project2/` — charts (`scaling_pandas_mapreduce_spark.png`, `budget_allocation.png`) + `run_log.txt`.

## What it does (maps to the brief)
| Brief requirement | In the script |
|---|---|
| Data: date, campaign, platform, impressions, clicks, spend, conversions, revenue, age, gender, city, device | `generate_data()` — simulated, 12 columns |
| Calculate CTR, CPC, CPA, conversion rate, ROAS, revenue, campaign performance | `add_metrics()`, `campaign_performance()` |
| Big Data simulation: 10k → 100k → 1M → 10M records | `scaling_simulation()` |
| Compare Pandas vs MapReduce vs Spark | three engines, **cross-checked for identical results** |
| Marketing Budget Recommendation Engine (e.g. FB 25% / IG 30% …) | `recommend_budget()` — ROAS-weighted split of Rs. 1,000,000 |

## Example result (simulated data, seed=42)
Recommended split of the next Rs. 1,000,000 (by historical ROAS):

| Platform | % | Rs. |
|---|---|---|
| Facebook | 23.3% | 233,301 |
| Google | 23.3% | 232,941 |
| Instagram | 19.2% | 191,555 |
| YouTube | 19.0% | 190,211 |
| TikTok | 15.2% | 151,991 |

Big-Data timing (aggregate by platform, same result all three ways) — **real Spark (pyspark 4.3 on JDK 21):**

| Rows | Pandas | MapReduce | Spark |
|---|---|---|---|
| 10k | 0.002s | 0.008s | 0.957s |
| 100k | 0.010s | 0.135s | 0.474s |
| 1M | 0.075s | 0.725s | 0.721s |
| 10M | 0.677s | 8.990s | 3.908s |

Pandas (vectorised C) wins on one machine; pure-Python MapReduce scales linearly but is slowest; Spark reads the data from disk and aggregates in the JVM — its start-up/IO overhead only pays off on a cluster with data too big for one machine's RAM. Note the crossover: by 10M rows Spark already beats pure-Python MapReduce.

## Note on Spark
This runs **real Spark** (pyspark 4.3, local mode) on **JDK 21** — the script
auto-detects an installed JDK 17/21 and sets `JAVA_HOME` (the machine's default
`java` is Java 27, which Spark does not support). Two Windows gotchas are handled
in code:

- **Paths with spaces** ("Big Data Analytics") break Hadoop's path parser, so Spark
  reads from a space-free staging dir under the temp folder.
- **PySpark 4.3 + Python 3.13** crashes the Spark *Python worker*, so the Spark path
  uses Spark's JVM-native CSV reader (explicit schema) + `collect()` and never
  routes data through a Python worker.

If no compatible JDK is found, the Spark stage automatically falls back to a
multiprocessing map→reduce stand-in (clearly labelled) so the script still runs.

## Honesty
The dataset is **simulated** (labelled as such), so this demonstrates the workflow,
not a real market claim. The budget split assumes past ROAS persists and ignores
diminishing returns, audience saturation and seasonality — validate with A/B tests
before committing real spend.

## Run
```bash
python project2_budget_optimizer.py
```
Exits non-zero if any requirement check fails (17 checks + chart files).
