import ast
import pickle
from pathlib import Path

import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
MOVIES_PATH = BASE_DIR / "tmdb_5000_movies.csv"
CREDITS_PATH = BASE_DIR / "tmdb_5000_credits.csv"
MOVIE_LIST_OUT = BASE_DIR / "movie_list.pkl"
SIMILARITY_OUT = BASE_DIR / "similarity.pkl"


def parse_names(text, top_n=None):
    items = []
    try:
        items = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError):
        return []

    names = [item.get("name", "") for item in items if isinstance(item, dict)]
    names = [name.replace(" ", "") for name in names if name]
    if top_n is not None:
        names = names[:top_n]
    return names


def parse_director(text):
    try:
        items = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError):
        return []

    for item in items:
        if isinstance(item, dict) and item.get("job") == "Director":
            name = item.get("name", "")
            if name:
                return [name.replace(" ", "")]
    return []


def stem_text(text):
    stemmer = PorterStemmer()
    return " ".join(stemmer.stem(word) for word in text.split())


def build_training_frame(movies_path, credits_path):
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    merged = movies.merge(credits, on="title")
    data = merged[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]].copy()
    data = data.dropna().drop_duplicates(subset=["title"]).reset_index(drop=True)

    data["genres"] = data["genres"].apply(parse_names)
    data["keywords"] = data["keywords"].apply(parse_names)
    data["cast"] = data["cast"].apply(lambda value: parse_names(value, top_n=3))
    data["crew"] = data["crew"].apply(parse_director)
    data["overview"] = data["overview"].apply(lambda value: value.split())

    data["tags"] = (
        data["overview"]
        + data["genres"]
        + data["keywords"]
        + data["cast"]
        + data["crew"]
    )

    data["tags"] = data["tags"].apply(lambda tokens: " ".join(tokens).lower())
    data["tags"] = data["tags"].apply(stem_text)

    return data


def compute_similarity_matrices(data):
    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectors = vectorizer.fit_transform(data["tags"])
    content_similarity = cosine_similarity(vectors)

    # Collaborative option from metadata co-occurrence (no user ratings available in TMDB files).
    collab_vectorizer = CountVectorizer(max_features=5000, stop_words="english", binary=True)
    collab_vectors = collab_vectorizer.fit_transform(
        (data["genres"].apply(lambda x: " ".join(x)) + " " + data["cast"].apply(lambda x: " ".join(x)) + " " + data["crew"].apply(lambda x: " ".join(x)))
    )
    collaborative_similarity = cosine_similarity(collab_vectors)

    return {
        "content": content_similarity,
        "collaborative": collaborative_similarity,
    }


def main():
    data = build_training_frame(MOVIES_PATH, CREDITS_PATH)
    similarity = compute_similarity_matrices(data)

    movie_list = data[["movie_id", "title"]].reset_index(drop=True)

    with open(MOVIE_LIST_OUT, "wb") as file:
        pickle.dump(movie_list, file)

    with open(SIMILARITY_OUT, "wb") as file:
        pickle.dump(similarity, file)

    print(f"Saved: {MOVIE_LIST_OUT.name}, {SIMILARITY_OUT.name}")
    print(f"Movies processed: {len(movie_list)}")


if __name__ == "__main__":
    main()
