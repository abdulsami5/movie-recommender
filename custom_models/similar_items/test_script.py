import pandas as pd

from custom_models.similar_items.recommender import TFIDFRecommender
from custom_models.utils import get_top_matched_titles, get_movie_by_id

movies_df = pd.read_csv('../data/imdb_processed.csv')
recommender = TFIDFRecommender(movies_df)
recommender.prepare_data()
recommender.fit()

recommender.save_model("movie_recommender.pkl")
recommender = TFIDFRecommender.load_model('movie_recommender.pkl')


while True:
    title = input("Search Title: ")
    ids = get_top_matched_titles(movies_df, title)
    for id in ids:
        print(get_movie_by_id(movies_df, id))

    test_movie_id = int(input("Movie ID for recommendation: "))
    print(get_movie_by_id(movies_df, test_movie_id))
    print("Recommendations:")
    for movie_id in recommender.recommend_similar_items(test_movie_id):
        print(get_movie_by_id(movies_df, movie_id))


# new_movie = {
#     'TITLE': 'Avengers Reassemble',
#     'GENRES': 'Action|Sci-Fi|Adventure',
#     'DESCRIPTION': 'Earth’s mightiest heroes reunite to face a cosmic threat bigger than ever before.'
# }
#
# for item in recommender.recommend_for_new_item(new_movie):
#     print(get_movie_by_id(movies_df, item))
#
#
# while True:
#     title = input("Title: ")
#     print(title)
#     ids = get_top_matched_titles(pd.read_csv('../data/imdb_processed.csv'), title)
#     for id in ids:
#         print(get_movie_by_id(movies_df, id))