import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
import part_1.part_1 as p1


# Tuned hyperparameters over the validation set
N_FACTORS = 36
N_EPOCHS = 40
LEARNING_RATE = 0.00335
REGULARIZATION = 0.0183
RANDOM_SEED = 42


def _build_trainset(ratings: pd.DataFrame):
    """
    Build a Surprise trainset from the ratings.

    Surprise's SVD runs SGD over the ratings in trainset order and never
    shuffles them. ratings.dat is sorted by user, so without a shuffle every
    epoch would update one user many times in a row. The ratings are
    shuffled once with the fixed seed so the update order is reproducible.
    """
    rating_columns = ["user_id", "movie_id", "rating"]
    shuffled = ratings[rating_columns].sample(frac=1, random_state=RANDOM_SEED)

    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(shuffled, reader).build_full_trainset()


def fit_matrix_factorization(ratings: pd.DataFrame) -> SVD:
    """
    Fit an SVD model with the tuned hyperparameters on the given ratings.

    The fixed seed initialises P and Q the same way on every fit.
    """
    model = SVD(
        n_factors=N_FACTORS,
        n_epochs=N_EPOCHS,
        biased=True,
        lr_all=LEARNING_RATE,
        reg_all=REGULARIZATION,
        random_state=RANDOM_SEED,
    )
    model.fit(_build_trainset(ratings))
    return model


def matrix_factorization_model(
        ratings_train: pd.DataFrame,
        ratings_validation: pd.DataFrame,
        movies: pd.DataFrame,
) -> SVD:
    """
    Fit a biased matrix factorization model with SGD (Surprise SVD).

    The model minimises, over the observed ratings only,

        sum (r(u,i) - mu - b_u - b_i - p_u . q_i)^2
            + lambda * (||p_u||^2 + ||q_i||^2 + b_u^2 + b_i^2)

    Despite its name, Surprise's SVD computes no singular value
    decomposition: it is Funk-style MF fitted by SGD (Lecture 2.1, slides 6-9).

    Fit on train and evaluate on validation, then refit on train plus
    validation, so test ratings stay unseen and later history is used.
    """
    print("\n========== MATRIX FACTORIZATION (SVD, SGD) ==========")
    print(f"Factors: {N_FACTORS}")
    print(f"Epochs: {N_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Regularization: {REGULARIZATION}")
    print(f"Random seed: {RANDOM_SEED}")

    model = fit_matrix_factorization(ratings_train)

    # Validation for choosing the hyperparameters, so test stays unbiased.
    # SGD minimises squared error, so RMSE shows whether the fit worked;
    # NDCG@10 shows whether it ranks well (Lecture 2.1, slide 11).
    validation_pairs = ratings_validation[["user_id", "movie_id", "rating"]]
    rating_predictions = model.test(validation_pairs.itertuples(index=False))
    p1.print_prediction_quality("Validation", rating_predictions)

    recommendations_valid = matrix_factorization_recommend(
        model=model,
        ratings_history=ratings_train,
        user_ids=ratings_validation["user_id"].unique(),
        catalogue=movies["movie_id"],
        top_n=10,
    )
    print("\n========== Matrix factorization (validation, all users) ==========")
    p1.metric_evaluation(
        predictions=recommendations_valid,
        ground_truth=ratings_validation,
        catalogue=movies["movie_id"],
        k=10,
        relevance_threshold=4,
    )

    # Refit on train + validation
    combined_ratings = pd.concat([ratings_train, ratings_validation], ignore_index=True)
    return fit_matrix_factorization(combined_ratings)


def matrix_factorization_recommend(
        model: SVD,
        ratings_history: pd.DataFrame,
        user_ids,
        catalogue: pd.Series,
        top_n: int = 10,
) -> pd.DataFrame:
    """
    Recommend the top-n unseen catalogue movies per user with the fitted MF.

    Every candidate movie i is scored with

        score(u, i) = mu + b_u + b_i + p_u . q_i

    For known users this matches model.predict (without its clipping to the
    rating scale), but scores the whole catalogue with one matrix product
    instead of one call per pair.

    Cold-start users, whom the model never saw, are scored with model.predict,
    which falls back to mu + b_i. Their ranking is therefore by item bias, a
    popularity-like ranking that is the same for every cold-start user.
    Movies the model never saw are not candidates.
    The output has the same columns as recommend_popular:
    user_id, movie_id, rank, score.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    trainset = model.trainset

    # Include every historical rating, including low ratings.
    seen_by_user = (
        ratings_history.groupby("user_id")["movie_id"]
        .agg(set)
        .to_dict()
    )

    # Candidates are the catalogue movies the model knows, ordered by
    # movie_id so that a stable sort breaks score ties by movie_id.
    inner_by_movie = {
        trainset.to_raw_iid(inner_id): inner_id
        for inner_id in trainset.all_items()
    }
    candidate_movies = np.array(
        sorted(set(catalogue) & inner_by_movie.keys())
    )
    candidate_inner = np.array(
        [inner_by_movie[movie_id] for movie_id in candidate_movies],
        dtype=int,
    )
    candidate_factors = model.qi[candidate_inner]
    candidate_bias = model.bi[candidate_inner]

    inner_by_user = {
        trainset.to_raw_uid(inner_id): inner_id
        for inner_id in trainset.all_users()
    }

    # model.predict gives every unknown user the same scores, so they are
    # computed once, on the first cold-start user, and reused.
    cold_start_scores = None

    recommendations = []

    for user_id in pd.Series(user_ids).unique():
        inner_uid = inner_by_user.get(user_id)
        if inner_uid is None:
            if cold_start_scores is None:
                cold_start_scores = np.array([
                    model.predict(user_id, movie_id).est
                    for movie_id in candidate_movies
                ])
            scores = cold_start_scores.copy()
        else:
            scores = (
                trainset.global_mean
                + model.bu[inner_uid]
                + candidate_bias
                + candidate_factors @ model.pu[inner_uid]
            )

        seen_movies = seen_by_user.get(user_id, set())
        scores[np.isin(candidate_movies, list(seen_movies))] = -np.inf

        best = np.argsort(-scores, kind="stable")[:top_n]
        best = best[np.isfinite(scores[best])]

        recommendations.append(
            pd.DataFrame(
                {
                    "user_id": user_id,
                    "movie_id": candidate_movies[best],
                    "rank": np.arange(1, len(best) + 1),
                    "score": scores[best],
                }
            )
        )

    columns = ["user_id", "movie_id", "rank", "score"]

    if not recommendations:
        return pd.DataFrame(columns=columns)

    return pd.concat(recommendations, ignore_index=True)[columns]


def sparsity_test(
        test_users,
        shuffled_history: pd.DataFrame,
        ground_truth: pd.DataFrame,
        catalogue: pd.Series,
) -> list[dict]:
    """
    Refit the MF on 50% and 20% of the shuffled training interactions and
    evaluate on the full test set. Hyperparameters stay fixed.

    Taking a prefix of one shuffled ordering makes the samples nested
    (20% is inside 50%). The subset is also the user history, so seen
    movies are filtered against what the model was trained on. Test users
    who lose all their history become cold-start users and get the
    item-bias ranking from model.predict.
    """
    results = []

    for fraction in (0.50, 0.20):
        sample_size = int(len(shuffled_history) * fraction)
        sampled_history = shuffled_history.iloc[:sample_size]

        model = fit_matrix_factorization(sampled_history)
        sparse_recommendations = matrix_factorization_recommend(
            model=model,
            ratings_history=sampled_history,
            user_ids=test_users,
            catalogue=catalogue,
            top_n=10,
        )

        n_cold_start = len(set(test_users) - set(sampled_history["user_id"]))
        print(f"\n========== Matrix factorization test, {fraction:.0%} history ==========")
        print(f"Training interactions: {sample_size}")
        print(f"Cold-start test users (bias-only ranking): {n_cold_start}")
        metrics = p1.metric_evaluation(
            predictions=sparse_recommendations,
            ground_truth=ground_truth,
            catalogue=catalogue,
            k=10,
            relevance_threshold=4,
        )

        results.append({
            "training_fraction": fraction,
            "training_interactions": sample_size,
            "metrics": metrics,
        })

    return results
