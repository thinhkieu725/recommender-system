# part_2/markov_chain.py
"""
This file has first and second order Markov chain implementations for movie recommendation. 
It:
- First get the list of positively rated movies for each user.
- Then fit Markov models based on these movie watching sequences.
- And generate recommendations based on the fitted models. 

This version only uses positive ratings (>= 4) to build sequences and transitions, ignoring low ratings like 1, 2, and 3.
"""


import pandas as pd


POSITIVE_THRESHOLD = 4 # Ratings >= 4 are considered positive ratings for building sequences and transitions.
DEFAULT_RECENT_N = 5 


def build_positive_sequences(ratings: pd.DataFrame, positive_threshold: int = POSITIVE_THRESHOLD) -> dict[int, list[int]]:

    """
    This function builds chronological movie sequences containing only positively rated movies.

    Low ratings are removed before transitions are constructed. 
    The intention is that two positive ratings can form a transition even if a low rating occurred
    between them in the original interaction history.

    Input:
    - ratings: DataFrame with columns ["user_id", "movie_id", "rating", "timestamp"]

    Output:
    - user_watch_sequences: Dictionary mapping user_id to a list of movie_ids, sorted by timestamp
    - example: {1: [A, B, C], 2: [D, E], 3: [F]} 

    """

    required = {"user_id", "movie_id", "rating", "timestamp"}
    if not required.issubset(ratings.columns):
        raise ValueError(f"ratings must contain {required}")

    positively_rated_movies = ratings.loc[
        ratings["rating"] >= positive_threshold,
        ["user_id", "movie_id", "timestamp"],
    ].copy()

    positively_rated_movies = positively_rated_movies.sort_values(
        ["user_id", "timestamp", "movie_id"]
    )

    user_watch_sequences = {}

    for user_id, group in positively_rated_movies.groupby("user_id"):
        movie_ids = group["movie_id"].astype(int).tolist()
        user_watch_sequences[user_id] = movie_ids

    return user_watch_sequences

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

def fit_markov_models(ratings_train: pd.DataFrame, positive_threshold: int = POSITIVE_THRESHOLD) -> dict:
    """
    Estimate first- and second-order transition probabilities from training
    data only.

    First order:  P(next_item | previous_item)
    Second order: P(next_item | previous_item_1, previous_item_2)

    Input:
    - ratings_train: DataFrame with columns ["user_id", "movie_id", "rating", "timestamp"]

    Output:
    - models: Dictionary containing:
        + "watch_sequences": Dictionary mapping user IDs to their positive watch sequences
        + "first_order":     Dictionary mapping each movie to the probabilities of its next movies
        + "second_order":    Dictionary mapping each pair of movies to the probabilities of their next movies
    """

    # Get sequences of positively rated movies for each user first
    positive_sequences = build_positive_sequences(
        ratings_train,
        positive_threshold=positive_threshold,
    )

    # Count first- and second-order transitions
    first_order_counts = {}     # Example: {A: {C: 3, E: 1}, B: {D: 2}}
    second_order_counts = {}    # Example: {(A, B): {C: 2}, (B, C): {D: 1, E: 1}}

    for sequence in positive_sequences.values():
        for i in range(1, len(sequence)):
            previous_item = sequence[i - 1]
            next_item = sequence[i]

            if previous_item not in first_order_counts:
                first_order_counts[previous_item] = {}

            next_item_counts = first_order_counts[previous_item]

            # if next_item is new, .get() returns 0, then we add 1 to make the count 1, otherwise we +1 to the existing count.
            next_item_counts[next_item] = next_item_counts.get(next_item, 0) + 1 

        for i in range(2, len(sequence)):
            item_pair = (sequence[i - 2], sequence[i - 1])
            next_item = sequence[i]
            if item_pair not in second_order_counts:
                second_order_counts[item_pair] = {}

            next_item_counts = second_order_counts[item_pair]

            # same logic
            next_item_counts[next_item] = next_item_counts.get(next_item, 0) + 1

    return {
        "watch_sequences": positive_sequences,
        "first_order": _normalise_counts(first_order_counts),   # Example: {A: {C: 0.75, E: 0.25}, B: {D: 1.0}}
        "second_order": _normalise_counts(second_order_counts), # Example: {(A, B): {C: 1.0}, (B, C): {D: 0.5, E: 0.5}}
    }


def _score_first_order(movie_history: list[int], transition_probabilities: dict) -> dict[int, float]:
    """
    This function averages next-item probabilities across the selected history.

    How it works:
    1. Take each movie in the user's movie history (with omitted low ratings).
    2. Retrieve the movie's learned next-item probabilities (probabilities of what the user might like next).
    3. Adds up those probabilities for each candidate movie.
    4. Divides by the number of watched movies that provided predictions
    5. Returns a dictionary of candidate movies and their averaged scores.

    Input:
    - movie_history: List of movie_ids representing the user's positive rating history
        + example: [A, B, C, D]

    - transition_probabilities: Dictionary mapping previous_item to a dictionary of next-item probabilities
        + example: {A: {C: 0.75, E: 0.25}, B: {D: 1.0}} 

    Output:
    - scores: Dictionary mapping movie_id to its averaged score based on the history
        + example: {C: 0.375, E: 0.125, D: 0.5}
    """
    scores = {}
    used_movies = 0 # Count of movies that have learned transitions

    for previous_item in movie_history:
        next_items = transition_probabilities.get(previous_item) # {C: 0.75, E: 0.25}, if exist of course

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


def recommend_markov(
    models: dict,
    ratings_history: pd.DataFrame,
    user_ids: list[int],
    catalogue: pd.Series,
    order: int = 1,                   
    history_mode: str = "full",       
    recent_n: int = DEFAULT_RECENT_N, 
    top_n: int = 10,                  
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
    - recent_n:        Number of recent positive items to consider if history_mode is "recent" 
    - top_n:           Number of top recommendations to return for each user

    Output:
    - recommendations: DataFrame with columns ["user_id", "movie_id", "rank", "score"]

    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    
    if history_mode not in ("full", "recent"):
        raise ValueError("history_mode must be 'full' or 'recent'.")
    
    if recent_n < 1 or top_n < 1:
        raise ValueError("recent_n and top_n must be at least 1.")

    catalogue_ids = pd.Series(catalogue).drop_duplicates().astype(int).tolist() # List of all unique movie IDs in the catalogue
    catalogue_set = set(catalogue_ids) # Set of all unique movie IDs in the catalogue for faster lookup

    # Get sequences of positively rated movies for each user
    positive_sequences = build_positive_sequences(ratings_history)

    
    seen_by_user = (
        ratings_history.groupby("user_id")["movie_id"]
        .agg(set)
        .to_dict()
    )

    first_order = models["first_order"]
    second_order = models["second_order"]

    recommendations = []

    for user_id in pd.Series(user_ids).unique():
        history = positive_sequences.get(user_id, [])

        if history_mode == "recent":
            history = history[-recent_n:] # Use only the most recent N positively rated movies for scoring

        if order == 1:
            scores = _score_first_order(history, first_order)
        else:
            scores = _score_second_order(history, second_order)

            # Back off to first order when no second-order context was learned.
            if not scores:
                scores = _score_first_order(history, first_order)

        seen_movies = seen_by_user.get(user_id, set())

        candidates = [
            (movie_id, score)
            for movie_id, score in scores.items()
            if movie_id in catalogue_set and movie_id not in seen_movies
        ]

        # If no learned transition can score a candidate, leave the user's
        # recommendation list empty rather than inventing a preference.
        candidates.sort(key=lambda x: (-x[1], x[0])) # -x[1] for descending score, x[0] for ascending movie_id in case of tie
        candidates = candidates[:top_n]

        for rank, (movie_id, score) in enumerate(candidates, start=1):
            recommendations.append(
                {
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "rank": rank,
                    "score": score,
                }
            )

    return pd.DataFrame(
        recommendations,
        columns=["user_id", "movie_id", "rank", "score"],
    )