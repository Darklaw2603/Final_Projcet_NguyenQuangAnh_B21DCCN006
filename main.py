import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from recommender import recommend_movies
from collaborative import train_collaborative_model, recommend_for_user
from recommender import get_similar_movies
from hybrid import hybrid_recommendation


# ===== LOAD DATA =====
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

print("Data shape:", df.shape)

# ===== CLEANING =====
df = df.drop_duplicates(subset=["userId", "movieId"])

# lọc user & movie ít rating
df = df[df["userId"].isin(df["userId"].value_counts()[lambda x: x >= 5].index)]
df = df[df["movieId"].isin(df["movieId"].value_counts()[lambda x: x >= 5].index)]

# ===== EDA =====
sns.histplot(df["rating"], bins=10)
plt.title("Phân bố rating")
plt.show()

# ===== TẠO BẢNG PHIM RIÊNG =====
movies_df = movies.copy()

# ===== TEST RECOMMENDER =====
recommend_movies("Toy Story (1995)", movies_df)

# ===== COLLABORATIVE FILTERING =====
print("\n🔹 Huấn luyện Collaborative Filtering (SVD)...")

svd_model, rmse, mae = train_collaborative_model(df)

print(f"📊 RMSE: {rmse:.4f}")
print(f"📊 MAE : {mae:.4f}")

# ===== RECOMMEND FOR USER =====
recommend_for_user(
    model=svd_model,
    user_id=1,
    movies_df=movies,
    ratings_df=ratings,
    top_n=5
)

# ===== HYBRID RECOMMENDATION =====
similar_movies = get_similar_movies(
    movie_title="Toy Story (1995)",
    movies_df=movies,
    top_n=20
)

hybrid_recommendation(
    user_id=1,
    movie_title="Toy Story (1995)",
    svd_model=svd_model,
    movies_df=movies,
    ratings_df=ratings,
    similar_movies=similar_movies,
    top_n=5,
    alpha=0.5
)
