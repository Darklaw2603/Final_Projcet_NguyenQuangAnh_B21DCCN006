import streamlit as st
import pandas as pd

from recommender import get_similar_movies
from collaborative import train_collaborative_model, recommend_for_user
from hybrid import hybrid_recommendation


st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.markdown("**Content-Based + Collaborative + Hybrid Recommendation**")

# ===== LOAD DATA =====
@st.cache_data
def load_data():
    movies = pd.read_csv(
        "data/movies.dat",
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1"
    )

    ratings = pd.read_csv(
        "data/ratings.dat",
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"]
    )

    df = ratings.merge(movies, on="movieId")
    df = df.drop_duplicates(subset=["userId", "movieId"])

    df = df[df["userId"].isin(df["userId"].value_counts()[lambda x: x >= 5].index)]
    df = df[df["movieId"].isin(df["movieId"].value_counts()[lambda x: x >= 5].index)]

    return movies, ratings, df


movies, ratings, df = load_data()

# ===== TRAIN SVD =====
@st.cache_resource
def train_model(df):
    model, rmse, mae = train_collaborative_model(df)
    return model, rmse, mae


svd_model, rmse, mae = train_model(df)

# ===== SIDEBAR =====
st.sidebar.header("⚙️ Tùy chọn")

user_id = st.sidebar.number_input(
    "Chọn User ID",
    min_value=int(ratings["userId"].min()),
    max_value=int(ratings["userId"].max()),
    value=1
)

movie_title = st.sidebar.selectbox(
    "Chọn phim yêu thích",
    sorted(movies["title"].unique())
)

alpha = st.sidebar.slider(
    "Hybrid weight (Content ↔ Collaborative)",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

# ===== METRICS =====
st.subheader("📊 Đánh giá mô hình")
col1, col2 = st.columns(2)
col1.metric("RMSE", f"{rmse:.4f}")
col2.metric("MAE", f"{mae:.4f}")

# ===== CONTENT-BASED =====
st.subheader("🎞 Content-Based Recommendation")

similar_movies = get_similar_movies(
    movie_title=movie_title,
    movies_df=movies,
    top_n=5
)

for mid, score in similar_movies:
    title = movies[movies["movieId"] == mid]["title"].values[0]
    st.write(f"- {title} (Similarity: {score:.3f})")

# ===== HYBRID =====
st.subheader("🔀 Hybrid Recommendation")

hybrid_movies = get_similar_movies(
    movie_title=movie_title,
    movies_df=movies,
    top_n=20
)

results = []
watched = ratings[ratings["userId"] == user_id]["movieId"].unique()

for movie_id, content_score in hybrid_movies:
    if movie_id in watched:
        continue

    svd_score = svd_model.predict(user_id, movie_id).est
    hybrid_score = alpha * content_score + (1 - alpha) * (svd_score / 5)

    title = movies[movies["movieId"] == movie_id]["title"].values[0]
    results.append((title, hybrid_score, content_score, svd_score))

results.sort(key=lambda x: x[1], reverse=True)

for r in results[:5]:
    st.write(
        f"🎬 **{r[0]}** | "
        f"Hybrid: `{r[1]:.3f}` | "
        f"Content: `{r[2]:.3f}` | "
        f"SVD: `{r[3]:.2f}`"
    )
