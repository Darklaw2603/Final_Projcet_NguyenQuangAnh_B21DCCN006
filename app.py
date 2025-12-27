import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.markdown("**Content-Based • Collaborative Filtering (Surprise) • Hybrid**")

# =========================
# LOAD DATA
# =========================
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

    # Cleaning
    df = df.drop_duplicates(subset=["userId", "movieId"])
    df = df.dropna()

    return movies, ratings, df


movies, ratings, df = load_data()

# =========================
# CONTENT-BASED MODEL
# =========================
@st.cache_resource
def build_content_model(movies_df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_df["genres"])
    cosine_sim = cosine_similarity(tfidf_matrix)
    indices = pd.Series(movies_df.index, index=movies_df["title"])
    return cosine_sim, indices


cosine_sim, indices = build_content_model(movies)


def get_similar_movies(title, top_n=5):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    return [(movies.iloc[i]["movieId"], score) for i, score in sim_scores]


# =========================
# COLLABORATIVE FILTERING (SURPRISE)
# =========================
@st.cache_resource
def train_svd(df):
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(
        df[["userId", "movieId", "rating"]],
        reader
    )

    trainset, testset = train_test_split(
        data, test_size=0.2, random_state=42
    )

    model = SVD(n_factors=100, random_state=42)
    model.fit(trainset)

    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    return model, rmse, mae


svd_model, rmse, mae = train_svd(df)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Cấu hình")

user_id = st.sidebar.selectbox(
    "Chọn User ID",
    sorted(df["userId"].unique())
)

movie_title = st.sidebar.selectbox(
    "Chọn phim yêu thích",
    sorted(movies["title"].unique())
)

alpha = st.sidebar.slider(
    "Hybrid weight (Content ↔ Collaborative)",
    0.0, 1.0, 0.5
)

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs([
    "🎯 Gợi ý phim",
    "📊 Phân tích & Trực quan hóa",
    "📈 Đánh giá mô hình"
])

# =========================================================
# TAB 1 – RECOMMENDATION
# =========================================================
with tab1:
    st.subheader("🎞 Content-Based Recommendation")

    for mid, score in get_similar_movies(movie_title):
        title = movies[movies["movieId"] == mid]["title"].values[0]
        st.write(f"- {title} (Similarity: {score:.3f})")

    st.subheader("🔀 Hybrid Recommendation")

    watched = df[df["userId"] == user_id]["movieId"].values
    results = []

    for mid, content_score in get_similar_movies(movie_title, top_n=20):
        if mid in watched:
            continue

        svd_score = svd_model.predict(user_id, mid).est / 5
        hybrid_score = alpha * content_score + (1 - alpha) * svd_score

        title = movies[movies["movieId"] == mid]["title"].values[0]
        results.append((title, hybrid_score, content_score, svd_score))

    results.sort(key=lambda x: x[1], reverse=True)

    for r in results[:5]:
        st.write(
            f"🎬 **{r[0]}** | Hybrid: `{r[1]:.3f}` | "
            f"Content: `{r[2]:.3f}` | SVD: `{r[3]:.2f}`"
        )

# =========================================================
# TAB 2 – DATA ANALYSIS & VISUALIZATION (YÊU CẦU MỤC 3)
# =========================================================
with tab2:
    st.subheader("📌 1. Phân bố rating (Histogram)")

    fig, ax = plt.subplots()
    sns.histplot(df["rating"], bins=10, kde=True, ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("📌 2. Tần suất nhóm sản phẩm (Genres)")

    genres_count = (
        df["genres"]
        .str.get_dummies("|")
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots()
    genres_count.plot(kind="bar", ax=ax)
    ax.set_ylabel("Number of Movies")
    st.pyplot(fig)

    st.subheader("📌 3. Top Items (phim được rating nhiều nhất)")

    top_movies = (
        df.groupby("title")["rating"]
        .count()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots()
    top_movies.plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Ratings")
    st.pyplot(fig)

    st.subheader("📌 4. Heatmap User–Movie (subset)")

    sample_users = df["userId"].unique()[:20]
    sample_movies = df["movieId"].unique()[:20]

    pivot = df[
        df["userId"].isin(sample_users) &
        df["movieId"].isin(sample_movies)
    ].pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# =========================================================
# TAB 3 – EVALUATION
# =========================================================
with tab3:
    st.subheader("📈 Đánh giá mô hình Collaborative Filtering (Surprise SVD)")

    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{rmse:.4f}")
    col2.metric("MAE", f"{mae:.4f}")

    st.markdown("""
    **Chỉ số đánh giá**
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)

    Mô hình được huấn luyện bằng **Surprise SVD** trên MovieLens dataset.
    """)
