import ast
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MOVIE_LIST_PATH = BASE_DIR / "movie_list.pkl"
SIMILARITY_PATH = BASE_DIR / "similarity.pkl"
MOVIES_CSV_PATH = BASE_DIR / "tmdb_5000_movies.csv"


def parse_genres(text):
    try:
        items = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError):
        return "N/A"

    names = [item.get("name", "") for item in items if isinstance(item, dict) and item.get("name")]
    return ", ".join(names[:3]) if names else "N/A"


@st.cache_resource
def load_artifacts():
    with open(MOVIE_LIST_PATH, "rb") as file:
        movie_list = pickle.load(file)

    with open(SIMILARITY_PATH, "rb") as file:
        similarity = pickle.load(file)

    return movie_list, similarity


@st.cache_data
def load_movie_details():
    details = pd.read_csv(MOVIES_CSV_PATH, usecols=["id", "title", "overview", "genres"])
    details["genres"] = details["genres"].apply(parse_genres)
    return details


def recommend(movie, movie_list, similarity_payload, details_df, method="content"):
    title_to_index = pd.Series(movie_list.index, index=movie_list["title"]).to_dict()

    if movie not in title_to_index:
        return []

    matrix = similarity_payload.get(method)
    if matrix is None:
        matrix = similarity_payload.get("content")

    movie_index = title_to_index[movie]
    distances = list(enumerate(matrix[movie_index]))
    distances = sorted(distances, key=lambda item: item[1], reverse=True)

    top_indices = [idx for idx, _ in distances[1:6]]
    recommended = movie_list.iloc[top_indices].copy()

    merged = recommended.merge(details_df, left_on=["movie_id", "title"], right_on=["id", "title"], how="left")
    merged = merged[["movie_id", "title", "overview", "genres"]]
    merged["overview"] = merged["overview"].fillna("Overview not available")
    merged["genres"] = merged["genres"].fillna("N/A")

    return merged.to_dict(orient="records")


def main():
    st.set_page_config(page_title="Movie Recommender", layout="wide")
    st.title("Content-Based Movie Recommendation System")

    movie_list, similarity_payload = load_artifacts()
    details_df = load_movie_details()

    selected_movie = st.selectbox("Select a movie", movie_list["title"].tolist())
    method = st.radio(
        "Recommendation method",
        options=["content", "collaborative"],
        format_func=lambda value: "Content-Based" if value == "content" else "Collaborative",
        horizontal=True,
    )

    if st.button("Recommend", type="primary"):
        recommendations = recommend(
            movie=selected_movie,
            movie_list=movie_list,
            similarity_payload=similarity_payload,
            details_df=details_df,
            method=method,
        )

        if not recommendations:
            st.warning("No recommendations available for this title.")
            return

        cols = st.columns(5)
        for idx, rec in enumerate(recommendations):
            with cols[idx]:
                st.subheader(rec["title"])
                st.caption(f"Movie ID: {rec['movie_id']}")
                st.write(f"Genres: {rec['genres']}")
                st.write(rec["overview"][:220] + ("..." if len(rec["overview"]) > 220 else ""))


if __name__ == "__main__":
    main()
