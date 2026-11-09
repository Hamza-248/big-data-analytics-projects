"""
Project 3 - Pakistan Social Audience & Campaign Intelligence Engine
Streamlit web application answering the brief's five marketing questions.

Run:
    streamlit run project3_app.py

Data is SIMULATED and clearly labelled. Example comments are illustrative, not
real measurements from any brand. Real use must follow each platform's Terms.
"""

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import project3_pipeline as pipe

st.set_page_config(page_title="PK Social Audience Engine", page_icon="🧵",
                   layout="wide")


@st.cache_data(show_spinner=False)
def load(n_posts, seed):
    return pipe.build_everything(n_posts=n_posts, seed=seed)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🧵 Audience & Campaign Engine")
st.sidebar.caption("Clustering-based intelligence for a Pakistani lawn-suit brand")
n_posts = st.sidebar.slider("Number of posts (simulated)", 200, 1000, 400, 50)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42, 1)
platform_f = st.sidebar.multiselect("Filter platform", pipe.PLATFORMS, pipe.PLATFORMS)
st.sidebar.warning("Data is **simulated** for demonstration. Example comments are "
                   "illustrative, not real brand measurements. Follow each "
                   "platform's Terms of Service for real data.")

R = load(n_posts, int(seed))
posts = R["posts"][R["posts"]["platform"].isin(platform_f)].copy()
comments = R["comments"][R["comments"]["post_id"].isin(posts["post_id"])].copy()

st.title("Pakistan Social Audience & Campaign Intelligence Engine")
st.markdown("A clustering-based engine that **discovers audience segments and "
            "campaign opportunities** from engagement data and bilingual "
            "(English / Roman-Urdu / Urdu) comments — not just sentiment.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Posts", len(posts))
c2.metric("Comments", len(comments))
c3.metric("Need segments", comments["topic"].nunique())
c4.metric("Anomalies flagged", int(R["anomalies"]["is_anomaly"].sum()))

tabs = st.tabs([
    "1 · Content that attracts",
    "2 · What the audience wants",
    "3 · Unusual patterns",
    "4 · Audience needs",
    "5 · What to do next",
    "Data",
])

# ---------------------------------------------------------------- Q1
with tabs[0]:
    st.header("Which content attracts attention?")
    cp = pipe.content_performance(posts)
    st.caption("Compared on **median** engagement per post (not totals), so a "
               "content type isn't ranked just for having more posts.")
    chart = (alt.Chart(cp).mark_bar().encode(
        x=alt.X("median_likes:Q", title="Median likes per post"),
        y=alt.Y("content_type:N", sort="-x", title=None),
        tooltip=["content_type", "posts", "median_likes", "median_comments",
                 "median_shares", "median_views"],
        color=alt.value("#3b5bdb")))
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(cp.round(3), use_container_width=True, hide_index=True)
    best = cp.iloc[0]
    st.success(f"**{best['content_type']}** leads with a median of "
               f"{int(best['median_likes'])} likes across {int(best['posts'])} posts.")

# ---------------------------------------------------------------- Q2
with tabs[1]:
    st.header("What does the audience want?")
    ts = pipe.topic_summary(comments)
    st.caption("Comments grouped by need (price, delivery, discount, quality, "
               "sizing, availability, purchase interest, praise, complaint).")
    chart = (alt.Chart(ts).mark_bar().encode(
        x=alt.X("comments:Q", title="Number of comments"),
        y=alt.Y("need_segment:N", sort="-x", title=None),
        tooltip=["need_segment", "comments", "pct", "positive", "negative"],
        color=alt.value("#2f9e44")))
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(ts[["need_segment", "comments", "pct", "positive", "negative"]],
                 use_container_width=True, hide_index=True)

    st.subheader("Unsupervised check: do these needs emerge from the text alone?")
    cm = R["cluster_metrics"]
    a, b = st.columns(2)
    a.metric("Comment clusters (KMeans, char n-grams)", cm["k"])
    b.metric("Silhouette / topic purity",
             f"{cm['silhouette']:.2f} / {cm['purity']:.2f}")
    st.caption("Short bilingual comments are genuinely hard to cluster, so purity "
               "is partial — production would combine clustering with keyword "
               "rules. The engine uses the interpretable topic grouping above for "
               "decisions and clustering as a discovery/validation layer.")
    st.markdown("**Language mix of comments:**")
    lang = comments["language_type"].value_counts().rename_axis("language").reset_index(name="comments")
    st.altair_chart(alt.Chart(lang).mark_arc(innerRadius=50).encode(
        theta="comments:Q", color=alt.Color("language:N",
        scale=alt.Scale(range=["#3b5bdb", "#f08c00", "#2f9e44"])),
        tooltip=["language", "comments"]), use_container_width=True)

# ---------------------------------------------------------------- Q3
with tabs[2]:
    st.header("Which marketing patterns are unusual?")
    an = pipe.detect_anomalies(posts)
    st.caption("IsolationForest flags posts whose engagement differs substantially "
               "from comparable posts — for human review, not automatic conclusions.")
    flagged = an[an["is_anomaly"]].copy()
    sc = (alt.Chart(an).mark_circle(size=70, opacity=0.6).encode(
        x=alt.X("views:Q", title="Views"),
        y=alt.Y("likes:Q", title="Likes"),
        color=alt.Color("is_anomaly:N", title="Anomaly",
                        scale=alt.Scale(domain=[False, True],
                                        range=["#adb5bd", "#e03131"])),
        tooltip=["post_id", "content_type", "platform", "likes", "comments_count",
                 "shares", "views", "anomaly_score"]))
    st.altair_chart(sc, use_container_width=True)
    st.subheader(f"{len(flagged)} posts flagged for investigation")
    st.dataframe(flagged[["post_id", "platform", "content_type", "campaign",
                          "likes", "comments_count", "shares", "views",
                          "anomaly_score"]].round(3),
                 use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- Q4
with tabs[3]:
    st.header("Which audience needs should the brand address?")
    ts = pipe.topic_summary(comments)
    st.markdown("Named need-segments discovered from the comments:")
    for _, row in ts.iterrows():
        with st.expander(f"**{row['need_segment']}** — {int(row['comments'])} "
                         f"comments ({row['pct']}%)"):
            ex = comments[comments["topic"] == row["topic"]]["comment_text"].head(5).tolist()
            st.write("Example comments:")
            for e in ex:
                st.markdown(f"> {e}")

    st.subheader("Engagement segments (post clustering)")
    post_seg, seg_profile = pipe.cluster_posts(posts)
    st.dataframe(seg_profile[["segment_name", "posts", "likes", "comments_count",
                              "shares", "views"]].round(0),
                 use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- Q5
with tabs[4]:
    st.header("What should the marketing team do next?")
    cp = pipe.content_performance(posts)
    ts = pipe.topic_summary(comments)
    an = pipe.detect_anomalies(posts)
    recs = pipe.recommendations(posts, comments, cp, ts, an)
    for _, r in recs.iterrows():
        st.markdown(f"#### {r['area']}")
        st.markdown(f"- **Finding:** {r['finding']}")
        st.markdown(f"- **Action:** {r['action']}")
    st.info("Recommendations are evidence-linked but based on **simulated** data; "
            "validate with controlled A/B tests before committing budget.")

# ---------------------------------------------------------------- Data
with tabs[5]:
    st.header("Underlying data (simulated)")
    st.subheader("Table A — Social media posts")
    st.dataframe(posts.drop(columns=["_seeded_anomaly"], errors="ignore").head(50),
                 use_container_width=True, hide_index=True)
    st.subheader("Table B — Comments")
    st.dataframe(comments.head(50), use_container_width=True, hide_index=True)
    st.download_button("Download posts CSV",
                       posts.drop(columns=["_seeded_anomaly"], errors="ignore").to_csv(index=False),
                       "project3_posts.csv", "text/csv")
    st.download_button("Download comments CSV", comments.to_csv(index=False),
                       "project3_comments.csv", "text/csv")
