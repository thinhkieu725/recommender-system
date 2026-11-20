# part_2/markov_chain.py
"""
Build rating histories for each user, estimate first- and second-order transition models, 
and generate recommendations. Histories can group ratings with the same timestamp; 
transition counts and recommendation scores can be rating-weighted.

"""


import pandas as pd
import math

DEFAULT_RECENT_N = 5 


def build_rating_sequences(
        ratings: pd.DataFrame,
        group_same_timestamp: bool = False,
    ) -> dict[int, list[list[tuple[int, float]]]]:

    """
    This function builds chronological rating sequences for every user.

    Each sequence is a list of "steps". A step is a list of (movie_id, rating) pairs.
    - group_same_timestamp=False: every step holds exactly one rating (each rating is a separate step).
    - group_same_timestamp=True:  ratings sharing the same timestamp form one step, so
      their relative order (which would otherwise be decided by movie_id) is not used.

    Input:
    - ratings: DataFrame with columns ["user_id", "movie_id", "rating", "timestamp"]

    Output:
    - user_watch_sequences: Dictionary mapping user_id to a list of steps, sorted by timestamp
    - example (no grouping): {1: [[(A, 5.0)], [(B, 4.0)], [(C, 2.0)]]}
    - example (grouping, A and B rated in the same second): {1: [[(A, 5.0), (B, 4.0)], [(C, 2.0)]]}

    """

    required = {"user_id", "movie_id", "rating", "timestamp"}
    if not required.issubset(ratings.columns):
        raise ValueError(f"ratings must contain {required}")

    data = ratings.sort_values(
        ["user_id", "timestamp", "movie_id"]
    ).copy()

    user_watch_sequences = {}

    for user_id, group in data.groupby("user_id"):
        steps = []
        previous_timestamp = None

        for movie_id, rating, timestamp in zip(
            group["movie_id"], group["rating"], group["timestamp"]
        ):
            # Timestamp grouping:
            # If enabled, ratings with the same timestamp are grouped into one step, 
            # so no transitions are inferred between those ratings
            if group_same_timestamp and steps and timestamp == previous_timestamp:
                steps[-1].append((int(movie_id), float(rating)))
            else:
                steps.append([(int(movie_id), float(rating))])
            
            previous_timestamp = timestamp

        user_watch_sequences[user_id] = steps

    return user_watch_sequences

def rating_weight(rating: float, method: str = "linear") -> float:
    """
    This function assigns a weight to a rating based on the specified method.

    Input:
    - rating: The rating value (float)
    - method: The weighting method to use ("linear" or "exponential" or "positive")

    Output:
    - weight: The computed weight for the given rating.
    """

    if method == "linear":
        return (rating - 1.0) / 4.0
    
    elif method == "exponential":
        alpha = 0.5
        return math.exp(alpha * (rating - 3.0)) 
    
    elif method == "positive":
        return max(0.0, rating - 3.0)

    else:
        raise ValueError(f"Unknown weighting method: {method}")

    
def _normalise_counts(counts: dict) -> dict:
    """
    This function converts each movie's next-movie counts into probabilities.

    Input:
    - counts: Dictionary mapping movie_id/movie-pair to a dictionary of next-movie counts
    - example:

        For first-order counts: {A: {C: 3, E: 1}, B: {D: 2}}
        
        meaning: Movie A was followed by movie C three times and movie E once;
                 movie B was followed by movie D twice.

        For second-order counts: {(A, B): {C: 2}, (B, C): {D: 1, E: 1}}

        meaning: The sequence (movie A, movie B) was followed by movie C twice;
                 the sequence (movie B, movie C) was followed by movie D once and movie E once.

    Output:
    - probabilities: Dictionary mapping movie_id/movie-pair to a dictionary of next-movie probabilities
    - example:
        For first-order probabilities: {A: {C: 0.75, E: 0.25}, B: {D: 1.0}}
        
        meaning: Given movie A, the probability of next movie being C is 0.75 and E is 0.25;
                 given movie B, the probability of next movie being D is 1.0.

        For second-order probabilities: {(A, B): {C: 1.0}, (B, C): {D: 0.5, E: 0.5}}

        meaning: Given the sequence (movie A, movie B), the probability of next movie being C is 1.0;
                 given the sequence (movie B, movie C), the probability of next movie being D is 0.5 and E is 0.5.

        Sum of probabilities for each movie_id/movie-pair will be 1.0.

    """
    probabilities = {}

    # Here, "context" is either a single movie_id (for first-order) or a tuple of two movie_ids (for second-order).
    for context, next_movies_counts in counts.items():
        total = sum(next_movies_counts.values())

        if total:
            probabilities[context] = {}

            for movie_id, count in next_movies_counts.items():
                probabilities[context][movie_id] = count / total

    return probabilities

def fit_markov_models(
        ratings_train: pd.DataFrame, 
        weight_method: str = "linear",
        weighting: str = "destination",
        group_same_timestamp: bool = False,
    ) -> dict:
    """
    Estimate first- and second-order transition probabilities from training
    data only.

    First order:  P(next_item | previous_item)
    Second order: P(next_item | previous_item_1, previous_item_2)

    Input:
    - ratings_train: DataFrame with columns ["user_id", "movie_id", "rating", "timestamp"]
    - group_same_timestamp: if True, ratings with the same timestamp form one step and
      no transitions are learned between them (see build_rating_sequences)
    - weight_method: "linear", "exponential" or "positive" (see rating_weight)
    - weighting: "none", "destination" or "both" (see get_transition_weight)

    Output:
    - models: Dictionary containing:
        + "watch_sequences": Dictionary mapping user IDs to their rating sequences
        + "first_order":     Dictionary mapping each movie to the probabilities of its next movies
        + "second_order":    Dictionary mapping each pair of movies to the probabilities of their next movies
    """

    if weighting not in ("none","destination", "both"):
        raise ValueError("Invalid weighting method. Choose from 'none', 'destination', or 'both'.")

    # Build rating sequences for each user, optionally grouping ratings with the same timestamp
    sequences = build_rating_sequences(
        ratings_train,
        group_same_timestamp=group_same_timestamp,
    )

    # Count first- and second-order transitions
    first_order_counts = {}     # Example: {A: {C: 3, E: 1}, B: {D: 2}}
    second_order_counts = {}    # Example: {(A, B): {C: 2}, (B, C): {D: 1, E: 1}}

    def get_transition_weight(source_rating, destination_rating):
        if weighting == "none":
            return 1.0

        destination_weight = rating_weight(
            destination_rating, weight_method
        )

        if weighting == "destination":
            return destination_weight

        source_weight = rating_weight(
            source_rating, weight_method    
        )

        return source_weight * destination_weight
    
    for steps in sequences.values():

        # First-order transitions
        # Without grouping every step has one rating, so this is the same as looping over consecutive ratings.
        # with grouping, each movie in one step can transition to movies in the next step.
        for t in range(1, len(steps)):

            # Timestamp grouping:
            # Splitting the weight across the next step 
            # stops one large same-second batch from dominating the counts (share = 1 without grouping).
            share = 1.0 / len(steps[t])
            

            for previous_item, previous_rating in steps[t - 1]:
                for next_item, next_rating in steps[t]:

                    weight = get_transition_weight(previous_rating, next_rating) * share

                    if weight == 0.0:
                        continue

                    if previous_item not in first_order_counts:
                        first_order_counts[previous_item] = {}

                    counts = first_order_counts[previous_item]

                    counts[next_item] = counts.get(next_item, 0.0) + weight

        # Second-order transitions
        for t in range(2, len(steps)):

            share = 1.0 / len(steps[t])  # --- IMPROVEMENT 3 (see above) ---

            for first_item, _ in steps[t - 2]:
                for second_item, source_rating in steps[t - 1]:
                    item_pair = (first_item, second_item)

                    for next_item, next_rating in steps[t]:

                        weight = get_transition_weight(source_rating, next_rating) * share

                        if weight == 0.0:
                            continue

                        if item_pair not in second_order_counts:
                            second_order_counts[item_pair] = {}

                        counts = second_order_counts[item_pair]

                        # Add this weighted transition count to the current second-order context
                        counts[next_item] = counts.get(next_item, 0.0) + weight

    return {
        "watch_sequences": sequences,
        "first_order": _normalise_counts(first_order_counts),   # Example: {A: {C: 0.75, E: 0.25}, B: {D: 1.0}}
        "second_order": _normalise_counts(second_order_counts), # Example: {(A, B): {C: 1.0}, (B, C): {D: 0.5, E: 0.5}}
        "weight_method": weight_method,
        "weighting": weighting,
        "group_same_timestamp": group_same_timestamp,  #  reused at recommend time 
    }


# weighting by the user's rating at scoring time 
def _source_weight(rating: float, weight_method: str | None) -> float:
    """
    How much a history movie counts when scoring.

    weight_method=None gives every history movie weight 1 (original behaviour).
    Otherwise the user's own rating is turned into a weight with rating_weight(),
    so a movie the user disliked contributes little or nothing.
    """
    if weight_method is None:
        return 1.0
    return rating_weight(rating, weight_method)

def _score_first_order(
        history_steps: list[list[tuple[int, float]]],
        transition_probabilities: dict,
        weight_method: str | None = None,
    ) -> dict[int, float]:
    """
    This function computes a weighted average of next-item probabilities across the selected history.

    How it works:
    1. Take each movie in the user's history.
    2. Retrieve the movie's learned next-item probabilities (probabilities of what the user might like next).
    3. Multiply them by the history movie's weight and add them up for each candidate movie.
    4. Divide by the total weight of the history movies that provided predictions.
    5. Return a dictionary of candidate movies and their averaged scores.

    With weight_method=None every weight is 1, which is a plain average (original behaviour).

    Input:
    - history_steps: The user's history as steps of (movie_id, rating) pairs
        + example: [[(A, 5.0)], [(B, 4.0)]]

    - transition_probabilities: Dictionary mapping previous_item to a dictionary of next-item probabilities
        + example: {A: {C: 0.75, E: 0.25}, B: {D: 1.0}}

    Output:
    - scores: Dictionary mapping movie_id to its averaged score based on the history
        + example (unweighted): {C: 0.375, E: 0.125, D: 0.5}
    """
    scores = {}
    total_weight = 0.0  # Sum of weights of movies that have learned transitions

    for step in history_steps:
        for previous_item, previous_rating in step:
            next_items = transition_probabilities.get(previous_item) # {C: 0.75, E: 0.25}, if exist of course

            if not next_items:
                continue

            # weight by the user's rating of this history movie 
            weight = _source_weight(previous_rating, weight_method)
            if weight == 0.0:
                continue

            total_weight += weight
            for movie_id, probability in next_items.items():
                scores[movie_id] = scores.get(movie_id, 0.0) + weight * probability

    if total_weight:
        for movie_id in scores:
            scores[movie_id] /= total_weight

    return scores

# Unused, newer version is _score_second_order_with_backoff() below
def _score_second_order(movie_history: list[int], transition_probabilities: dict) -> dict[int, float]:
    """
    This function averages next-item probabilities across consecutive history pairs.

    Same working logic as _score_first_order, but it uses pairs of consecutive movies 
    in the history to look up next-item probabilities.
    
    Input:
    - movie_history: List of movie_ids representing the user's positive rating history
      + example: [A, B, C, D]

    - transition_probabilities: Dictionary mapping (previous_item_1, previous_item_2) to a dictionary of next-item probabilities
      + example: {(A, B): {C: 1.0}, (B, C): {D: 0.5, E: 0.5}}

    Output:
    - scores: Dictionary mapping movie_id to its averaged score based on the history
      + example: {C: 0.5, D: 0.25, E: 0.25}
    
    """
    scores = {}
    used_movies = 0

    for i in range(1, len(movie_history)):
        context = (movie_history[i - 1], movie_history[i])
        next_items = transition_probabilities.get(context)

        if not next_items:
            continue

        used_movies += 1
        for movie_id, probability in next_items.items():

            if movie_id not in scores:
                scores[movie_id] = 0.0

            scores[movie_id] += probability

    if used_movies:
        for movie_id in scores:
            scores[movie_id] /= used_movies

    return dict(scores)

def _score_second_order_with_backoff(
    history_steps: list[list[tuple[int, float]]],
    first_order: dict,
    second_order: dict,
    weight_method: str | None = None,
) -> dict[int, float]:
    """
    Weighted average of next-item probabilities over consecutive (previous, current) pairs,
    falling back to the first-order row of `current` when the pair was never learned.

    With timestamp grouping, every movie in one step is paired with every movie in the next step.
    The weight of a pair is the user's rating of `current` (same choice as in fit_markov_models).
    """

    scores = {}
    total_weight = 0.0

    for t in range(1, len(history_steps)):
        for previous_item, _ in history_steps[t - 1]:
            for current_item, current_rating in history_steps[t]:

                context = (previous_item, current_item)

                next_items = second_order.get(context)

                # Fall back to first-order for this context only.
                if not next_items:
                    next_items = first_order.get(current_item)

                if not next_items:
                    continue

                # weight by the user's rating of the most recent movie in the pair
                weight = _source_weight(current_rating, weight_method)
                if weight == 0.0:
                    continue
            
                total_weight += weight

                for movie_id, probability in next_items.items():
                    scores[movie_id] = (
                        scores.get(movie_id, 0.0) + weight * probability
                    )

    if total_weight:
        scores = {
            movie_id: score / total_weight
            for movie_id, score in scores.items()
        }

    return scores


# timestamp grouping 
def _take_recent_steps(
        history_steps: list[list[tuple[int, float]]],
        recent_n: int,
    ) -> list[list[tuple[int, float]]]:
    """
    Keep the most recent steps until at least recent_n movies are covered.

    Without grouping this is exactly history[-recent_n:]. With grouping, a whole
    same-timestamp step is kept or dropped together, so the cut never depends on movie_id order.
    """
    selected = []
    count = 0

    for step in reversed(history_steps):
        if count >= recent_n:
            break
        selected.append(step)
        count += len(step)

    return selected[::-1]

def recommend_markov(
    models: dict,
    ratings_history: pd.DataFrame,
    user_ids: list[int],
    catalogue: pd.Series,
    order: int = 1,                   
    history_mode: str = "full",       
    recent_n: int = DEFAULT_RECENT_N,
    top_n: int = 10,
    score_weighting: str = "none",                  # "none" or "rating" 
    fallback_ranking: pd.DataFrame | None = None,   # e.g. p1.most_positively_rated(...) 
) -> pd.DataFrame:
    """
    Generate top-N recommendations using a first- or second-order Markov model.

    All previously rated movies, including negatively rated movies, are excluded from recommendations.

    Input:
    - models:          Dictionary containing fitted Markov models (from fit_markov_models)
    - ratings_history: DataFrame with columns ["user_id", "movie_id", "rating", "timestamp"]
    - user_ids:        List of user IDs to generate recommendations for
    - catalogue:       Series of movie IDs representing the catalogue of available movies
    - order:           1 for first-order Markov model, 2 for second-order Markov model
    - history_mode:    "full" or "recent" to determine how much of the user's history to use for scoring
    - recent_n:        Number of recent rated items to consider if history_mode is "recent" 
    - top_n:           Number of top recommendations to return for each user
    - score_weighting: "none" averages history movies equally; "rating" weights each history
                       movie by the user's own rating (using models["weight_method"])
    - fallback_ranking: Optional DataFrame with a "movie_id" column, best first. Used to fill
                       the list when Markov produces fewer than top_n candidates.

    Output:
    - recommendations: DataFrame with columns ["user_id", "movie_id", "rank", "score", "source"]
                       where source is "markov" or "fallback"

    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    
    if history_mode not in ("full", "recent"):
        raise ValueError("history_mode must be 'full' or 'recent'.")
    
    if recent_n < 1 or top_n < 1:
        raise ValueError("recent_n and top_n must be at least 1.")

    if score_weighting not in ("none", "rating"):
        raise ValueError("score_weighting must be 'none' or 'rating'.")

    catalogue_ids = pd.Series(catalogue).drop_duplicates().astype(int).tolist() # List of all unique movie IDs in the catalogue
    catalogue_set = set(catalogue_ids) # Set of all unique movie IDs in the catalogue for faster lookup

    # Get each user's rating sequence, grouped the same way the model was fitted
    rating_sequences = build_rating_sequences(
        ratings_history,
        group_same_timestamp=models.get("group_same_timestamp", False), 
    )


    seen_by_user = (
        ratings_history.groupby("user_id")["movie_id"]
        .agg(set)
        .to_dict()
    )

    first_order = models["first_order"]
    second_order = models["second_order"]

    # choose the scoring weight method 
    score_weight_method = models["weight_method"] if score_weighting == "rating" else None

    # popularity fallback, prepared once for all users 
    fallback_ids = []
    if fallback_ranking is not None:
        fallback_ids = [
            int(movie_id)
            for movie_id in fallback_ranking["movie_id"]
            if int(movie_id) in catalogue_set
        ]

    recommendations = []

    for user_id in pd.Series(user_ids).unique():
        history_steps = rating_sequences.get(user_id, [])

        if history_mode == "recent":
            # Use only the most recent N rated movies for scoring
            history_steps = _take_recent_steps(history_steps, recent_n) 

        if order == 1:
            scores = _score_first_order(history_steps, first_order, score_weight_method)
        else:
            scores = _score_second_order_with_backoff(
                history_steps, first_order, second_order, score_weight_method
            )

            # Back off to first order when no second-order context was learned.
            if not scores:
                scores = _score_first_order(history_steps, first_order, score_weight_method)

        seen_movies = seen_by_user.get(user_id, set())

        candidates = [
            (movie_id, score, "markov")
            for movie_id, score in scores.items()
            if movie_id in catalogue_set and movie_id not in seen_movies
        ]

        candidates.sort(key=lambda x: (-x[1], x[0])) # -x[1] for descending score, x[0] for ascending movie_id in case of tie
        candidates = candidates[:top_n]

        # popularity fallback
        # When Markov cannot fill top_n slots (cold start, or every candidate already seen),
        # fill the remaining slots with the most popular unseen movies, ranked after the Markov ones.
        if fallback_ids and len(candidates) < top_n:
            chosen = {movie_id for movie_id, _, _ in candidates}

            for movie_id in fallback_ids:
                if len(candidates) >= top_n:
                    break
                if movie_id in seen_movies or movie_id in chosen:
                    continue
                candidates.append((movie_id, 0.0, "fallback"))
                chosen.add(movie_id)
        

        for rank, (movie_id, score, source) in enumerate(candidates, start=1):
            recommendations.append(
                {
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "rank": rank,
                    "score": score,
                    "source": source,
                }
            )

    return pd.DataFrame(
        recommendations,
        columns=["user_id", "movie_id", "rank", "score", "source"],
    )