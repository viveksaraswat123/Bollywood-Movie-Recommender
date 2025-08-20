import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("BollywoodMovieDetail.csv")

# Fill missing values
for col in ['genre', 'actors', 'director', 'description']:
    if col in movies.columns:
        movies[col] = movies[col].fillna('')
    else:
        movies[col] = ""

# Combine features
movies['combined'] = movies['title'] + " " + movies['genre'] + " " + movies['actors'] + " " + movies['director'] + " " + movies['description']

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])

# Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function
def recommend(movie_title, top_n=5):
    if movie_title not in movies['title'].values:
        return []
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Streamlit UI
st.title("🎬 Bollywood Movie Recommendation System")
selected_movie = st.selectbox("Choose a movie:", movies['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    if recommendations:
        st.write("✅ Recommended Movies:")
        for r in recommendations:
            st.write("- " + r)
    else:
        st.write("⚠️ No recommendations found.")
