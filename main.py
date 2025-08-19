import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Load the dataset
df = pd.read_csv('BollywoodMovieDetail.csv')

# Fix column name and fill missing
df['genre'] = df['genre'].fillna('')

# Convert to lowercase for uniformity
df['genre'] = df['genre'].str.lower()

# Create count matrix from genre column, split by commas or spaces as needed
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
genre_matrix = vectorizer.fit_transform(df['genre'])

cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

def recommend(movie_title):
    idx = df[df['title'].str.lower() == movie_title.lower()].index
    if len(idx) == 0:
        print("Movie not found!")
        return
    idx = idx[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # top 5 recommendations excluding the movie itself

    print(f"Movies similar to '{movie_title}':")
    for i, score in sim_scores:
        print(df.iloc[i]['title'])

if __name__ == "__main__":
    while True:
        movie = input("\nEnter Bollywood movie title (or 'exit' to quit): ")
        if movie.lower() == 'exit':
            break
        recommend(movie)
