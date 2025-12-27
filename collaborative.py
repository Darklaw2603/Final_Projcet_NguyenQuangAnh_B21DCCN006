from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd


def train_collaborative_model(df):
    # Chỉ lấy các cột cần thiết
    data = df[["userId", "movieId", "rating"]]

    # Surprise cần format riêng
    reader = Reader(rating_scale=(0.5, 5.0))
    dataset = Dataset.load_from_df(data, reader)

    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

    # Model SVD
    model = SVD(random_state=42)
    model.fit(trainset)

    # Dự đoán
    predictions = model.test(testset)

    # Đánh giá
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    return model, rmse, mae

def recommend_for_user(model, user_id, movies_df, ratings_df, top_n=5):
    watched = ratings_df[ratings_df["userId"] == user_id]["movieId"].unique()
    all_movies = movies_df["movieId"].unique()

    candidates = [m for m in all_movies if m not in watched]

    predictions = []
    for movie_id in candidates:
        pred = model.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movies = predictions[:top_n]

    print(f"\n👤 Gợi ý phim cho User {user_id}:")
    for movie_id, score in top_movies:
        title = movies_df[movies_df["movieId"] == movie_id]["title"].values[0]
        print(f" - {title} (score: {score:.2f})")

