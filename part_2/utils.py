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
    """Compare models NDCG@10 minus baseline NDCG@10 on paired test users."""
    model_by_user = ranked_movies_by_user(model_recommendations)
    baseline_by_user = ranked_movies_by_user(baseline_recommendations)

    model_scores = []
    baseline_scores = []

    for user_id, ratings in ratings_test.groupby("user_id"):
        # Match the evaluator's eligibility rule.
        if not (ratings["rating"] >= 4).any():
            continue

        user_ratings = dict(zip(ratings["movie_id"], ratings["rating"]))
        model_scores.append(
            _ndcg_at_k(model_by_user.get(user_id, []), user_ratings, k=10)
        )
        baseline_scores.append(
            _ndcg_at_k(baseline_by_user.get(user_id, []), user_ratings, k=10)
        )

    if len(model_scores) < 2:
        raise ValueError("At least two eligible test users are required.")

    test = ttest_rel(model_scores, baseline_scores)
    interval = test.confidence_interval(confidence_level=0.95)

    model_mean = float(np.mean(model_scores))
    baseline_mean = float(np.mean(baseline_scores))

    return {
        "evaluated_users": len(model_scores),
        "model_ndcg": model_mean,
        "baseline_ndcg": baseline_mean,
        "mean_difference": model_mean - baseline_mean,
        "ci_95": (float(interval.low), float(interval.high)),
        "p_value": float(test.pvalue),
    }

def run_final_variablity(
        model_prediction: pd.DataFrame,
        kNN_result: pd.DataFrame,
        most_rated_result: pd.DataFrame,
        most_positive_result: pd.DataFrame,
        ground_truth: pd.DataFrame
) -> None:
    """ Run variability test for model against 3 baselines"""
    baselines = [
        ("Item-based kNN (all users)", kNN_result),
        ("Most-rated Item", most_rated_result),
        ("Most positive rated Item", most_positive_result)
    ]

    for model_name, baseline_prediction in baselines:
        print(f"\n========== {model_name.capitalize()} ==========")
        paired_results = compare_test_ndcg(
            model_recommendations = model_prediction,
            baseline_recommendations =  baseline_prediction,
            ratings_test = ground_truth
        )
        print(paired_results)