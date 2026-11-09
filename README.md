# Big Data Analytics — Course Projects

Three projects for the Big Data Analytics course, each self-contained with its
own code, dataset, outputs, and README.

| # | Project | Folder | Deliverable |
|---|---------|--------|-------------|
| 1 | Social Media Sentiment Analysis (EDA → cleaning → scaling → clustering) | [`project1-social-media-analysis/`](project1-social-media-analysis/) | Jupyter notebook + self-testing Python script |
| 2 | Digital Ad Budget Optimizer (pandas vs MapReduce vs **Apache Spark**) | [`project2-budget-optimizer/`](project2-budget-optimizer/) | Self-testing Python script |
| 3 | Pakistan Social Audience & Campaign Intelligence Engine (clustering web app) | *root of this repo* (see below) | **Live Streamlit app** |

## Project 3 — deployed web app

Project 3 is a Streamlit application and is **deployed live**:

### 🔗 https://big-data-analytics-project3.streamlit.app/

Its files live at the repository root so the hosting stays connected to
`project3_app.py` (the deployed main file):

- `project3_app.py` — the Streamlit web app (the deliverable)
- `project3_pipeline.py` — testable core (data generation, clustering, anomaly detection, recommendations)
- `project3_selftest.py` — headless requirement checks
- `requirements.txt` — dependencies used by Streamlit Community Cloud
- `README_project3.md` — full Project 3 documentation
- `data/project3_posts.csv`, `data/project3_comments.csv` — generated datasets

Run locally:

```bash
pip install -r requirements.txt
streamlit run project3_app.py
```

## Projects 1 & 2

Each folder is self-contained — the scripts resolve their `data/` and `outputs/`
folders relative to themselves, so they run from anywhere:

```bash
# Project 1
python project1-social-media-analysis/analysis.py

# Project 2
python project2-budget-optimizer/project2_budget_optimizer.py
```

## Note on data

All datasets for Projects 2 and 3 are **simulated and clearly labelled** as such.
Example comments in Project 3 are illustrative, not real brand measurements, and
brands are anonymised. Real deployment must follow each platform's Terms of Service.
