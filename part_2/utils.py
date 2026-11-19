import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from part_1.part_1 import _ndcg_at_k

#--- Variability report: statistical paired test---
def ranked_movies_by_user(recommendations: pd.DataFrame):
    """
    Reformat the DataFrame for compatibility with ndcg computation method
    """
    return (
        recommendations.sort_values(["user_id", "rank"])
        .groupby("user_id")["movie_id"]
        .agg(list)
        .to_dict()
    )
def compare_test_ndcg(model_recommendations,
                      baseline_recommendations,
                      ratings_test
    ):
    """Compare content-based minus baseline NDCG@10 on paired test users."""
    content_by_user = ranked_movies_by_user(model_recommendations)
    baseline_by_user = ranked_movies_by_user(baseline_recommendations)

    content_scores = []
    baseline_scores = []

    for user_id, ratings in ratings_test.groupby("user_id"):
        # Match the evaluator's eligibility rule.
        if not (ratings["rating"] >= 4).any():
            continue

        user_ratings = dict(zip(ratings["movie_id"], ratings["rating"]))
        content_scores.append(
            _ndcg_at_k(content_by_user.get(user_id, []), user_ratings, k=10)
        )
        baseline_scores.append(
            _ndcg_at_k(baseline_by_user.get(user_id, []), user_ratings, k=10)
        )

    if len(content_scores) < 2:
        raise ValueError("At least two eligible test users are required.")

    test = ttest_rel(content_scores, baseline_scores)
    interval = test.confidence_interval(confidence_level=0.95)

    content_mean = float(np.mean(content_scores))
    baseline_mean = float(np.mean(baseline_scores))

    return {
        "evaluated_users": len(content_scores),
        "content_ndcg": content_mean,
        "baseline_ndcg": baseline_mean,
        "mean_difference": content_mean - baseline_mean,
        "ci_95": (float(interval.low), float(interval.high)),
        "p_value": float(test.pvalue),
    }
