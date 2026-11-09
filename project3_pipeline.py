"""
Project 3 - Pakistan Social Audience & Campaign Intelligence Engine
===================================================================
CLUSTERING-BASED engine for a Lahore fashion brand (women's lawn suits) running
Instagram / TikTok / Facebook campaigns. It discovers audience segments and
campaign opportunities from engagement + bilingual (English / Roman-Urdu / Urdu)
comments, rather than only labelling sentiment.

This module is the testable core used by the Streamlit app (project3_app.py).
It answers the brief's five marketing questions:

  Q1  Which content attracts attention?      -> content-type performance
  Q2  What does the audience want?            -> comment topic clustering
  Q3  Which marketing patterns are unusual?   -> anomaly detection (IsolationForest)
  Q4  Which audience needs to address?        -> named need-segments
  Q5  What should marketing do next?          -> evidence-based recommendations

IMPORTANT: the data here is SIMULATED and clearly labelled as such. The example
comments in the brief are illustrative, not real measurements from any brand.
Real deployment must follow each platform's Terms of Service; do not assume
comments or analytics are publicly downloadable.
"""

import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SEED = 42

# --------------------------------------------------------------------------
# Reference vocabulary (brand-neutral, illustrative — NOT real brand data)
# --------------------------------------------------------------------------
BRANDS = ["Brand A", "Brand B", "Brand C"]        # anonymised stand-ins
PLATFORMS = ["Instagram", "TikTok", "Facebook"]
CAMPAIGNS = ["Eid Collection", "11.11 Sale", "12.12 Sale", "Summer Lawn", "Winter Edit"]
CONTENT_TYPES = ["Reel", "Product Photo", "Influencer Post",
                 "Styling Video", "Sale Announcement", "Eid Campaign"]
PRODUCT_CATEGORY = "Lawn suit"
LANGUAGES = ["English", "Roman Urdu", "Urdu"]

# Comment templates by TOPIC x LANGUAGE (short, realistic, bilingual).
COMMENT_BANK = {
    "price_inquiry": {
        "English": ["Price please?", "What's the price?", "How much is this?",
                    "Price of this suit?", "Can you share the price?"],
        "Roman Urdu": ["Price kya hai?", "Kitne ka hai?", "Price bata dein",
                       "Ye kitne ka hai?", "Rate kya hai bhai?"],
        "Urdu": ["قیمت کیا ہے؟", "یہ کتنے کا ہے؟", "ریٹ بتا دیں"],
    },
    "availability": {
        "English": ["Is this available in Karachi?", "In stock?",
                    "Is this available?", "Do you deliver to Lahore?"],
        "Roman Urdu": ["Stock hai?", "Karachi mein milega?", "Available hai abhi?",
                       "Islamabad delivery hoti hai?"],
        "Urdu": ["کیا یہ دستیاب ہے؟", "اسٹاک میں ہے؟"],
    },
    "delivery": {
        "English": ["Delivery time?", "How many days for delivery?",
                    "When will it arrive?", "Ordered last week, still waiting."],
        "Roman Urdu": ["Delivery kitne din mein hogi?", "Order kab tak aayega?",
                       "Pichle hafte order kiya tha, abhi tak nahi aaya."],
        "Urdu": ["ڈیلیوری کتنے دن میں ہوگی؟", "آرڈر کب آئے گا؟"],
    },
    "discount": {
        "English": ["Any discount on this suit?", "Is there a sale?",
                    "Do you have an offer?", "Discount available?"],
        "Roman Urdu": ["Koi discount hai?", "Sale kab lagegi?", "Discount milega?",
                       "Koi offer chal raha hai?"],
        "Urdu": ["کوئی رعایت ہے؟", "سیل کب لگے گی؟"],
    },
    "fabric_quality": {
        "English": ["Fabric quality?", "Is the lawn good quality?",
                    "How is the material?", "Is the cloth soft?"],
        "Roman Urdu": ["Fabric kaisa hai?", "Material acha hai?",
                       "Cloth ki quality kaisi hai?"],
        "Urdu": ["کپڑے کا معیار کیسا ہے؟", "میٹیریل اچھا ہے؟"],
    },
    "sizing": {
        "English": ["Available in large?", "Do you have XL?",
                    "Size chart please?", "Is medium available?"],
        "Roman Urdu": ["Size medium hai?", "Large milega?", "XL available hai?",
                       "Size chart bhej dein."],
        "Urdu": ["سائز میڈیم ہے؟", "لارج دستیاب ہے؟"],
    },
    "purchase_intent": {
        "English": ["How to order?", "I want to buy this.",
                    "Inbox me details.", "Please share order details."],
        "Roman Urdu": ["Order kaise karein?", "Mujhe ye chahiye.",
                       "Inbox me details send kar dein.", "Book karwana hai."],
        "Urdu": ["آرڈر کیسے کریں؟", "مجھے یہ خریدنا ہے۔"],
    },
    "praise": {
        "English": ["Very beautiful dress!", "Love this!", "Gorgeous!",
                    "So elegant!", "Beautiful colours."],
        "Roman Urdu": ["Bohat pyara hai!", "Zabardast!", "Khoobsurat design!",
                       "Bohat sohna hai."],
        "Urdu": ["بہت خوبصورت ہے!", "لاجواب ڈیزائن!"],
    },
    "complaint": {
        "English": ["Price is too high.", "Too expensive.", "Bad service.",
                    "Still waiting for my order."],
        "Roman Urdu": ["Price bohat zyada hai.", "Mehenga hai bohat.",
                       "Service achi nahi.", "Order abhi tak nahi mila."],
        "Urdu": ["قیمت بہت زیادہ ہے۔", "بہت مہنگا ہے۔"],
    },
}

# Topic -> (sentiment, purchase_intent) defaults
TOPIC_META = {
    "price_inquiry":   ("Neutral",  "Possible inquiry"),
    "availability":    ("Neutral",  "Possible inquiry"),
    "delivery":        ("Neutral",  "Existing buyer"),
    "discount":        ("Neutral",  "Price sensitive"),
    "fabric_quality":  ("Neutral",  "Researching"),
    "sizing":          ("Neutral",  "Researching"),
    "purchase_intent": ("Positive", "High intent"),
    "praise":          ("Positive", "Low / brand love"),
    "complaint":       ("Negative", "At risk"),
}

# Named audience need-segments the brand should address (Q4)
NEED_SEGMENTS = {
    "price_inquiry":   "Price-sensitive shoppers",
    "discount":        "Discount-focused commenters",
    "delivery":        "Delivery-concerned shoppers",
    "availability":    "Availability-asking customers",
    "fabric_quality":  "Quality-curious researchers",
    "sizing":          "Fit & sizing researchers",
    "purchase_intent": "Ready-to-buy customers",
    "praise":          "Brand advocates",
    "complaint":       "At-risk / unhappy customers",
}


# ==========================================================================
# DATA GENERATION  (simulated — clearly labelled)
# ==========================================================================
def generate_posts(n_posts=400, seed=SEED):
    rng = np.random.default_rng(seed)
    # per-content-type engagement multipliers (so content types really differ)
    ct_boost = {"Reel": 1.6, "Product Photo": 1.0, "Influencer Post": 1.9,
                "Styling Video": 1.4, "Sale Announcement": 1.2, "Eid Campaign": 2.1}
    ct = rng.choice(CONTENT_TYPES, n_posts)
    boost = np.array([ct_boost[c] for c in ct])
    base_likes = rng.lognormal(6.2, 0.6, n_posts) * boost
    likes = base_likes.astype(int) + 20
    comments = (likes * rng.uniform(0.05, 0.20, n_posts)).astype(int)
    shares = (likes * rng.uniform(0.01, 0.08, n_posts)).astype(int)
    views = (likes * rng.uniform(8, 25, n_posts)).astype(int)

    posts = pd.DataFrame({
        "post_id": [f"P{i:04d}" for i in range(1, n_posts + 1)],
        "brand": rng.choice(BRANDS, n_posts),
        "platform": rng.choice(PLATFORMS, n_posts),
        "campaign": rng.choice(CAMPAIGNS, n_posts),
        "content_type": ct,
        "product_category": PRODUCT_CATEGORY,
        "likes": likes,
        "comments_count": comments,
        "shares": shares,
        "views": views,
        "days_since_post": rng.integers(1, 90, n_posts),
    })

    # Inject a few genuine anomalies (viral / dead posts) for Q3 to find
    anom_idx = rng.choice(n_posts, size=max(5, n_posts // 40), replace=False)
    for i in anom_idx:
        if rng.random() < 0.5:                      # viral outlier: high everything
            posts.loc[i, ["likes", "comments_count", "shares", "views"]] *= rng.integers(6, 12)
        else:                                       # reach-without-engagement flop:
            # high views but engagement collapses -> a genuine multivariate outlier
            posts.loc[i, "views"] = int(posts.loc[i, "views"]) * rng.integers(4, 8)
            posts.loc[i, "likes"] = rng.integers(1, 10)
            posts.loc[i, ["comments_count", "shares"]] = 0
    posts["_seeded_anomaly"] = False
    posts.loc[anom_idx, "_seeded_anomaly"] = True
    return posts


def generate_comments(posts, avg_per_post=8, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    topics = list(COMMENT_BANK.keys())
    # topic popularity (price & availability dominate real fashion pages)
    topic_p = np.array([0.20, 0.16, 0.12, 0.12, 0.08, 0.08, 0.10, 0.09, 0.05])
    topic_p = topic_p / topic_p.sum()

    rows = []
    cid = 1
    for pid, ccount in zip(posts["post_id"], posts["comments_count"]):
        k = max(2, min(int(ccount), rng.poisson(avg_per_post) + 2))
        for _ in range(k):
            topic = rng.choice(topics, p=topic_p)
            lang = rng.choice(LANGUAGES, p=[0.45, 0.42, 0.13])
            text = rng.choice(COMMENT_BANK[topic][lang])
            sentiment, intent = TOPIC_META[topic]
            rows.append({
                "comment_id": f"C{cid:06d}",
                "post_id": pid,
                "comment_text": text,
                "language_type": lang,
                "topic": topic,                     # ground-truth label (for validation)
                "sentiment": sentiment,
                "purchase_intent": intent,
            })
            cid += 1
    return pd.DataFrame(rows)


# ==========================================================================
# Q1 · Which content attracts attention?
# ==========================================================================
def content_performance(posts):
    g = (posts.groupby("content_type")
         .agg(posts=("post_id", "size"),
              median_likes=("likes", "median"),
              median_comments=("comments_count", "median"),
              median_shares=("shares", "median"),
              median_views=("views", "median"))
         .sort_values("median_likes", ascending=False))
    # engagement rate = (likes+comments+shares)/views, guard views==0
    eng = posts.copy()
    eng["engagement_rate"] = ((eng["likes"] + eng["comments_count"] + eng["shares"])
                              / eng["views"].replace(0, np.nan))
    g["median_engagement_rate"] = (eng.groupby("content_type")["engagement_rate"]
                                    .median())
    return g.reset_index()


# ==========================================================================
# Q2 · What does the audience want?  (comment topic CLUSTERING)
# ==========================================================================
def cluster_comments(comments, k=9, seed=SEED):
    """Unsupervised TF-IDF + KMeans on comment_text. Returns comments with a
    'cluster' column, a cluster->label map (dominant known topic), and quality
    metrics (silhouette + purity vs the known topic labels)."""
    # char_wb n-grams bridge English / Roman-Urdu / Urdu (shared substrings),
    # which word tokens cannot — short bilingual comments cluster better this way.
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          max_features=600, min_df=2)
    X = vec.fit_transform(comments["comment_text"])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(X)
    out = comments.copy()
    out["cluster"] = labels

    # Label each cluster by its most common ground-truth topic
    cluster_topic = (out.groupby("cluster")["topic"]
                     .agg(lambda s: s.value_counts().index[0]).to_dict())
    out["cluster_label"] = out["cluster"].map(
        lambda c: NEED_SEGMENTS.get(cluster_topic[c], cluster_topic[c]))

    # Quality: silhouette (structure) + purity (agreement with known topics)
    try:
        sil = float(silhouette_score(X, labels)) if k > 1 else float("nan")
    except Exception:
        sil = float("nan")
    purity = float((out.groupby("cluster")["topic"]
                    .agg(lambda s: s.value_counts().iloc[0]).sum()) / len(out))
    return out, cluster_topic, {"silhouette": sil, "purity": purity, "k": k}


def topic_summary(comments):
    """Counts + language mix per discovered need, for Q2/Q4 tables."""
    s = (comments.groupby("topic")
         .agg(comments=("comment_id", "size"),
              positive=("sentiment", lambda x: (x == "Positive").sum()),
              negative=("sentiment", lambda x: (x == "Negative").sum()))
         .sort_values("comments", ascending=False))
    s["need_segment"] = [NEED_SEGMENTS.get(t, t) for t in s.index]
    s["pct"] = (s["comments"] / s["comments"].sum() * 100).round(1)
    return s.reset_index()


# ==========================================================================
# Q3 · Which marketing patterns are unusual?  (anomaly detection)
# ==========================================================================
def detect_anomalies(posts, contamination=0.05, seed=SEED):
    feats = ["likes", "comments_count", "shares", "views", "days_since_post"]
    X = StandardScaler().fit_transform(posts[feats])
    iso = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    pred = iso.fit_predict(X)                       # -1 = anomaly
    score = iso.score_samples(X)
    out = posts.copy()
    out["is_anomaly"] = (pred == -1)
    out["anomaly_score"] = score
    return out.sort_values("anomaly_score")


# ==========================================================================
# Q4 · Post engagement segments (clustering) + named audience needs
# ==========================================================================
def cluster_posts(posts, k=4, seed=SEED):
    feats = ["likes", "comments_count", "shares", "views"]
    Xs = StandardScaler().fit_transform(posts[feats])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    out = posts.copy()
    out["segment"] = km.fit_predict(Xs)
    profile = (out.groupby("segment")[feats].median())
    profile["posts"] = out["segment"].value_counts().sort_index()
    # name segments by engagement level (median likes rank)
    order = profile["likes"].rank(method="first").astype(int)
    names = {seg: lvl for seg, lvl in
             zip(profile.index, ["Low engagement", "Moderate engagement",
                                 "High engagement", "Viral / standout"])
             } if k == 4 else {}
    rank_to_name = dict(zip(profile.sort_values("likes").index,
                            ["Low engagement", "Moderate engagement",
                             "High engagement", "Viral / standout"][:k]))
    profile["segment_name"] = [rank_to_name.get(i, f"Segment {i}") for i in profile.index]
    out["segment_name"] = out["segment"].map(rank_to_name)
    return out, profile.reset_index()


# ==========================================================================
# Q5 · What should the marketing team do next?  (recommendations)
# ==========================================================================
def recommendations(posts, comments, content_perf, topic_sum, anomalies):
    recs = []
    # 1. Content: double down on the best content type by median engagement
    best_ct = content_perf.iloc[0]
    recs.append({
        "area": "Content",
        "finding": f"'{best_ct['content_type']}' posts have the highest median "
                   f"likes ({int(best_ct['median_likes'])}) across {int(best_ct['posts'])} posts.",
        "action": f"Produce more {best_ct['content_type']} content and A/B test it "
                  f"against Product Photos using median engagement per post.",
    })
    # 2. Audience need: biggest comment need-segment
    top_need = topic_sum.iloc[0]
    recs.append({
        "area": "Customer service",
        "finding": f"'{top_need['need_segment']}' is the largest comment segment "
                   f"({top_need['pct']}% of comments).",
        "action": "Add a pinned FAQ / auto-reply covering this need (e.g. price, "
                  "delivery, availability) in English and Roman Urdu to cut response time.",
    })
    # 3. Discount-focused share -> promotion strategy
    disc = topic_sum[topic_sum["topic"] == "discount"]
    disc_pct = float(disc["pct"].iloc[0]) if len(disc) else 0.0
    recs.append({
        "area": "Promotions",
        "finding": f"Discount-focused commenters are {disc_pct}% of comments.",
        "action": ("Test a limited bundle offer on lawn suits for the next campaign "
                   "and measure conversion, rather than blanket discounting."),
    })
    # 4. Purchase-intent -> sales follow-up
    intent = topic_sum[topic_sum["topic"] == "purchase_intent"]
    intent_pct = float(intent["pct"].iloc[0]) if len(intent) else 0.0
    recs.append({
        "area": "Sales",
        "finding": f"Ready-to-buy comments are {intent_pct}% of comments.",
        "action": "Route high-intent comments to DM/WhatsApp within an hour and "
                  "track reply-to-order conversion.",
    })
    # 5. Anomalies -> investigate
    n_anom = int(anomalies["is_anomaly"].sum())
    recs.append({
        "area": "Campaign testing",
        "finding": f"{n_anom} posts show unusual engagement vs comparable posts.",
        "action": "Review these posts: replicate what drove the viral outliers and "
                  "diagnose the dead ones (timing, creative, targeting).",
    })
    return pd.DataFrame(recs)


# ==========================================================================
# Orchestration (used by the app and the self-test)
# ==========================================================================
def build_everything(n_posts=400, seed=SEED, save=False):
    posts = generate_posts(n_posts, seed)
    comments = generate_comments(posts, seed=seed)
    content_perf = content_performance(posts)
    clustered, cluster_topic, cmetrics = cluster_comments(comments)
    topic_sum = topic_summary(comments)
    anomalies = detect_anomalies(posts)
    post_seg, seg_profile = cluster_posts(posts)
    recs = recommendations(posts, comments, content_perf, topic_sum, anomalies)
    if save:
        posts.drop(columns=["_seeded_anomaly"]).to_csv(
            os.path.join(DATA_DIR, "project3_posts.csv"), index=False)
        comments.to_csv(os.path.join(DATA_DIR, "project3_comments.csv"), index=False)
    return {
        "posts": posts, "comments": comments, "content_perf": content_perf,
        "clustered": clustered, "cluster_metrics": cmetrics, "topic_sum": topic_sum,
        "anomalies": anomalies, "post_seg": post_seg, "seg_profile": seg_profile,
        "recs": recs,
    }


if __name__ == "__main__":
    r = build_everything(save=True)
    print("posts:", len(r["posts"]), "comments:", len(r["comments"]))
    print("comment clusters metrics:", r["cluster_metrics"])
    print("anomalies:", int(r["anomalies"]["is_anomaly"].sum()))
    print("recommendations:", len(r["recs"]))
