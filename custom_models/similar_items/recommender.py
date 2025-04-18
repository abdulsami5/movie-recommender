import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#from custom_models.utils import get_movie_by_id, get_top_matched_titles


class TFIDFRecommender:
    known_genres = ['Action', 'Comedy', 'Thriller', 'Drama', 'Sci-Fi', 'Romance', 'Horror Movies', 'Animation']

    def __init__(self, df, title_weight=2, genres_weight=2, known_genre_weight=3):
        self.df = df.copy()
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

        self.title_weight = title_weight
        self.genres_weight = genres_weight
        self.known_genre_weight = known_genre_weight

        self.tfidf_matrix = None
        self.similarity_matrix = None

    def save_model(self, filepath='tfidf_model.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'similarity_matrix': self.similarity_matrix,
                'df': self.df,
                'title_weight': self.title_weight
            }, f)

    @classmethod
    def load_model(cls, filepath='tfidf_model.pkl'):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        recommender = cls(data['df'], title_weight=data['title_weight'])
        recommender.vectorizer = data['vectorizer']
        recommender.tfidf_matrix = data['tfidf_matrix']
        recommender.similarity_matrix = data['similarity_matrix']
        return recommender

    def boost_known_genres(self, genres):
        words = genres.split()
        boosted = []
        for word in words:
            if word in self.known_genres:
                boosted.extend([word] * self.known_genre_weight)  # boost known genres more
            else:
                boosted.append(word)
        return ' '.join(boosted)

    def prepare_data(self):
        self.df['GENRES'] = self.df['GENRES'].fillna('').str.replace('|', ' ')
        self.df['DESCRIPTION'] = self.df['DESCRIPTION'].fillna('')
        self.df['TITLE'] = self.df['TITLE'].fillna('')

        self.df['BOOSTED_GENRES'] = self.df['GENRES'].apply(self.boost_known_genres)

        self.df['TEXT'] = (
                (self.df['TITLE'] + ' ') * self.title_weight +
                (self.df['BOOSTED_GENRES'] + ' ') * self.genres_weight +
                self.df['DESCRIPTION']
        )

    def fit(self):

        self.prepare_data()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['TEXT'])
        # self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.similarity_matrix = None

    def get_similar_items(self, sim_scores, type_filter, top_n=10):
        # Add ITEM_ID for merging
        sim_scores = sim_scores.reset_index().rename(columns={0: 'score'})
        sim_scores['ITEM_ID'] = self.df.iloc[sim_scores['index']]['ITEM_ID'].values
        sim_scores['TYPE'] = self.df.iloc[sim_scores['index']]['TYPE'].values

        if type_filter:
            sim_scores = sim_scores[sim_scores['TYPE'] == type_filter]

        return sim_scores.sort_values('score', ascending=False).head(top_n)['ITEM_ID'].tolist()

    def recommend_similar_items(self, item_id, top_n=10, type_filter=None):
        if item_id not in self.df['ITEM_ID'].values:
            return []

        idx = self.df.index[self.df['ITEM_ID'] == item_id][0]

        # Get similarity scores for this item
        sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sim_scores = pd.Series(sim_scores)

        # Exclude itself
        sim_scores = sim_scores.drop(idx)

        return self.get_similar_items(sim_scores, type_filter, top_n)

    def recommend_for_new_item(self, new_item_dict, top_n=10, type_filter=None):
        text = (
                ((new_item_dict.get('TITLE', '') + ' ') * self.title_weight) +
                ((self.boost_known_genres(new_item_dict.get('GENRES', '') or '')) + ' ') * self.genres_weight +
                (new_item_dict.get('DESCRIPTION', '') or '')
        )

        new_vector = self.vectorizer.transform([text])
        sim_scores = cosine_similarity(new_vector, self.tfidf_matrix).flatten()
        sim_scores = pd.Series(sim_scores)
        return self.get_similar_items(sim_scores, type_filter, top_n)



