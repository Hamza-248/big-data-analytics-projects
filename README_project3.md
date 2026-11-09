# Project 3 — Pakistan Social Audience & Campaign Intelligence Engine (Clustering)

A small **web application** for a Lahore lawn-suit brand that **discovers audience
segments and campaign opportunities** from engagement data and bilingual
(English / Roman-Urdu / Urdu) comments — not just sentiment.

## Files
- `project3_app.py` — the Streamlit web app (the deliverable).
- `project3_pipeline.py` — testable core: data generation, clustering, anomaly detection, recommendations.
- `project3_selftest.py` — headless requirement checks (15 checks, exits non-zero on failure).
- `data/project3_posts.csv`, `data/project3_comments.csv` — generated datasets.

## The five marketing questions (app tabs)
| # | Question | How it's answered |
|---|---|---|
| 1 | Which content attracts attention? | Content-type comparison on **median** engagement (reels, product photos, influencer posts, styling videos, sale announcements, Eid campaigns) |
| 2 | What does the audience want? | Comments grouped by need (price, quality, sizing, delivery, discount, availability, purchase interest) + unsupervised **TF-IDF + KMeans** clustering as a discovery/validation layer |
| 3 | Which patterns are unusual? | **IsolationForest** anomaly detection on post engagement, for human review |
| 4 | Which audience needs to address? | Named need-segments (price-sensitive, discount-focused, delivery-concerned, availability-asking, ready-to-buy…) + post engagement **KMeans** segments |
| 5 | What should marketing do next? | Evidence-linked recommendations (content, service, promotions, sales, campaign testing) |

## Run
```bash
pip install streamlit scikit-learn pandas numpy altair
streamlit run project3_app.py
```
Then open http://localhost:8501. Use the sidebar to change the number of posts, the seed, or filter platforms.

Self-test (no server):
```bash
python project3_selftest.py
```
Result: **15/15 checks pass** (posts=400, comments≈3,950, anomalies≈20, comment-cluster purity≈0.45).

## Data tables
**Table A — posts:** post_id, brand, platform, campaign, content_type, product_category, likes, comments_count, shares, views, days_since_post.
**Table B — comments:** post_id, comment_text, language_type (English / Roman Urdu / Urdu), topic, sentiment, purchase_intent.

## Honesty / ethics (from the brief)
- The data is **simulated** and clearly labelled; the example comments are illustrative, **not** real measurements from Sapphire, Khaadi, Gul Ahmed or any brand (brands are anonymised as Brand A/B/C).
- Real deployment must follow each platform's **Terms of Service**; do not assume comments or analytics are publicly downloadable.
- Comment clustering of short bilingual text is genuinely hard (purity ≈ 0.45), so the engine uses interpretable topic grouping for decisions and clustering as a **discovery layer** — a hybrid a real team would use. Recommendations should be validated with A/B tests before spend.
- Real data sources to explore (check reuse permissions first): the 2026 paper *Sentiment Analysis of Social Media Data for Pakistani Fashion Brand Monitoring* (10,602 obs, 17 brands), Open Data Pakistan's e-commerce orders, and the Pakistani Fashion Catalogue on HuggingFace.
