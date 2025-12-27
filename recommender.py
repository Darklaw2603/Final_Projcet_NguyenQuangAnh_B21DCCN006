import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def recommend_movies(movie_title, movies_df, top_n=5):
    # tạo text feature
    movies_df = movies_df.copy()
    movies_df["text"] = movies_df["title"] + " " + movies_df["genres"]

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_df["text"])

    # cosine similarity trên PHIM (≈ 3900 x 3900)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()

    if movie_title not in indices:
        print("❌ Không tìm thấy phim")
        return

    idx = indices[movie_title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    movie_indices = [i[0] for i in sim_scores]

    print(f"\n🎬 Gợi ý phim tương tự '{movie_title}':")
    for title in movies_df.iloc[movie_indices]["title"]:
        print(" -", title)

def get_similar_movies(movie_title, movies_df, top_n=20):
    movies_df = movies_df.copy()
    movies_df["text"] = movies_df["title"] + " " + movies_df["genres"]

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_df["text"])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()

    if movie_title not in indices:
        return []

    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    return [(movies_df.iloc[i[0]]["movieId"], i[1]) for i in sim_scores]