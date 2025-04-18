from flask import Flask, request, render_template
import pandas as pd
from threading import Thread

from similar_items.recommender import TFIDFRecommender
from utils import get_top_matched_titles

app = Flask(__name__)

# Load movie data
movie_df = pd.read_csv("data/imdb_processed.csv")


# Initialize recommender
def load_model():
    global model
    print("Loading model in background...")
    model = TFIDFRecommender(movie_df)
    model.fit()


model = None
Thread(target=load_model).start()


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return render_template("index.html", results=[], query=query)

    matched_ids = get_top_matched_titles(movie_df, query)
    results = (
        movie_df[movie_df['ITEM_ID'].isin(matched_ids)]
        .loc[movie_df['ITEM_ID'].isin(matched_ids)]
        .copy()
    )
    results['order'] = pd.Categorical(results['ITEM_ID'], categories=matched_ids, ordered=True)
    results = results.sort_values('order')[['ITEM_ID', 'TITLE', 'GENRES']].to_dict(orient='records')
    return render_template("index.html", results=results, query=query)


@app.route("/recommend/<int:item_id>", methods=["GET"])
def recommend(item_id):
    similar_ids = model.recommend_similar_items(item_id, top_n=10)

    recommendations = (
        movie_df[movie_df['ITEM_ID'].isin(similar_ids)]
        .copy()
    )
    recommendations['order'] = pd.Categorical(recommendations['ITEM_ID'], categories=similar_ids, ordered=True)
    recommendations = recommendations.sort_values('order')[['ITEM_ID', 'TITLE', 'GENRES']].to_dict(orient='records')

    current_movie = movie_df[movie_df['ITEM_ID'] == item_id]['TITLE'].iloc[0]
    return render_template("index.html", movie_title=current_movie, recommendations=recommendations)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
    print(f"✅ Flask is starting on port 5000")
