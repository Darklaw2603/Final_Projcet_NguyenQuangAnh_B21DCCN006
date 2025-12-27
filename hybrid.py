def hybrid_recommendation(
    user_id,
    movie_title,
    svd_model,
    movies_df,
    ratings_df,
    similar_movies,
    top_n=5,
    alpha=0.5
):
    """
    alpha = 0.5 → cân bằng content & collaborative
    """

    watched = ratings_df[ratings_df["userId"] == user_id]["movieId"].unique()
    results = []

    for movie_id, content_score in similar_movies:
        if movie_id in watched:
            continue

        svd_score = svd_model.predict(user_id, movie_id).est

        # Hybrid score
        hybrid_score = alpha * content_score + (1 - alpha) * (svd_score / 5)

        title = movies_df[movies_df["movieId"] == movie_id]["title"].values[0]
        results.append((title, hybrid_score, content_score, svd_score))

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🔀 Hybrid gợi ý cho User {user_id} (từ '{movie_title}'):")
    for r in results[:top_n]:
        print(
            f" - {r[0]} | Hybrid: {r[1]:.3f} | "
            f"Content: {r[2]:.3f} | SVD: {r[3]:.2f}"
        )
