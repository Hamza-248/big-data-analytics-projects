# Social Media Sentiment Analysis — Course Project

Worked solution for the lecture walkthrough (Steps 5–9), run on the real dataset.

## Files
- `Social_Media_Analysis_Project.ipynb` — the analysis notebook (already executed, outputs + charts embedded).
- `data/social_media_sentiment_raw.csv` — dataset (2,260 rows × 13 columns) from
  [`mrudukp/social-media-sentiment-analysis`](https://github.com/mrudukp/social-media-sentiment-analysis).

## What each step does
| Step | Covered in notebook |
|------|---------------------|
| 5 · Data objects | One record = one **post**; dtypes, missing values, duplicates, column typing. |
| 6 · EDA | Platform/sentiment counts, median vs mean engagement, likes–comments relationship, posting-hour (valid timestamp). |
| 7 · Interpretation | Finding / Evidence / Action / Limitation, using **median + group size**, never total likes. |
| 8 · Scaling | Same analysis on 2k → 20k → 200k → 2M rows; records file size, load time, memory. |
| 9 · Analytics | Classification (sentiment), clustering (KMeans on scaled features), regression (likes), with train/test split. |

## Data cautions honoured (from the guide)
- **No `reach`/`impressions`** column → no engagement-rate-on-reach claims.
- `platform` case normalised (`Twitter` == `TWITTER`); 58 exact duplicates dropped; 8 impossible negative `likes` flagged.
- Blank `hashtags` treated as *"no hashtag,"* not missing data.
- Source is **not verified** as Pakistani brand data — used to teach the workflow, not to make market claims.

## Run it yourself
```bash
pip install pandas numpy scikit-learn matplotlib notebook
jupyter notebook Social_Media_Analysis_Project.ipynb
```
Or re-execute headless:
```bash
jupyter nbconvert --to notebook --execute --inplace Social_Media_Analysis_Project.ipynb
```
