"""
Project 3 - requirements self-test (headless, no server).
Run:  python project3_selftest.py
Exits 0 only if every requirement check passes.
"""
import os
import sys
import py_compile

import project3_pipeline as pipe

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

POST_COLS = ["post_id", "brand", "platform", "campaign", "content_type",
             "product_category", "likes", "comments_count", "shares", "views",
             "days_since_post"]
COMMENT_COLS = ["post_id", "comment_text", "language_type", "topic",
                "sentiment", "purchase_intent"]
REQUIRED_NEEDS = ["price_inquiry", "fabric_quality", "sizing", "delivery",
                  "discount", "availability", "purchase_intent"]


def main():
    print("=" * 70)
    print("PROJECT 3 · REQUIREMENTS CHECK")
    print("=" * 70)

    R = pipe.build_everything(n_posts=400, seed=42, save=True)
    posts, comments = R["posts"], R["comments"]
    cp, ts = R["content_perf"], R["topic_sum"]
    cm = R["cluster_metrics"]
    anoms, recs = R["anomalies"], R["recs"]

    checks = [
        ("Table A (posts) has all required fields",
         all(c in posts.columns for c in POST_COLS)),
        ("Table B (comments) has all required fields",
         all(c in comments.columns for c in COMMENT_COLS)),
        ("Comment volume is in the 2,000-5,000 demo range",
         2000 <= len(comments) <= 5000),
        ("Comments are tri-lingual (English / Roman Urdu / Urdu)",
         set(comments["language_type"].unique()) == {"English", "Roman Urdu", "Urdu"}),
        ("Product focus is a single category (lawn suit)",
         posts["product_category"].nunique() == 1),
        ("Q1: all 6 content types compared on median engagement",
         len(cp) == 6 and "median_likes" in cp.columns),
        ("Q2: comments grouped into the required need topics",
         all(t in set(comments["topic"]) for t in REQUIRED_NEEDS)),
        ("Q2: unsupervised comment clustering ran (silhouette + purity)",
         ("silhouette" in cm) and ("purity" in cm) and cm["k"] >= 5),
        ("Q3: anomaly detection flagged unusual posts",
         int(anoms["is_anomaly"].sum()) > 0),
        ("Q3: anomaly detector catches seeded outliers",
         anoms.loc[anoms["_seeded_anomaly"], "is_anomaly"].mean() >= 0.5),
        ("Q4: named audience need-segments produced",
         len(ts) >= 7 and "need_segment" in ts.columns),
        ("Q4: post engagement clustering produced segments",
         R["post_seg"]["segment_name"].nunique() >= 3),
        ("Q5: >=5 evidence-based recommendations with finding + action",
         len(recs) >= 5 and {"finding", "action", "area"}.issubset(recs.columns)),
        ("Data saved: posts + comments CSVs",
         os.path.exists(os.path.join(DATA_DIR, "project3_posts.csv"))
         and os.path.exists(os.path.join(DATA_DIR, "project3_comments.csv"))),
    ]

    # The Streamlit app must at least compile and its deps be importable.
    app_ok = True
    try:
        py_compile.compile(os.path.join(HERE, "project3_app.py"), doraise=True)
        import importlib.util
        app_ok = all(importlib.util.find_spec(m) is not None
                     for m in ("streamlit", "altair"))
    except Exception as e:
        app_ok = False
        print("   app compile/deps error:", str(e)[:120])
    checks.append(("Web app compiles and streamlit + altair are installed", app_ok))

    passed = 0
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        passed += bool(ok)
    print("-" * 70)
    print(f"  {passed}/{len(checks)} requirement checks PASSED")
    print(f"  (posts={len(posts)}, comments={len(comments)}, "
          f"anomalies={int(anoms['is_anomaly'].sum())}, "
          f"cluster purity={cm['purity']:.2f}, silhouette={cm['silhouette']:.2f})")
    all_ok = passed == len(checks)
    print("=" * 70)
    print("  RESULT:", "ALL REQUIREMENTS FULFILLED" if all_ok else "SOME REQUIREMENTS FAILED")
    print("=" * 70)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
