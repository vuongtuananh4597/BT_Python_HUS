import os
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

WORKSPACE = '/Users/vietnd/Documents/Datasets/'
DIR_MovieLens20M = os.path.join(WORKSPACE, 'ml-20m')
CUT_DATETIME = datetime(2013, 1, 1)
CUT_YEAR = 2013
CUT_TIMESTAMP = int((CUT_DATETIME - datetime(1970, 1, 1)).total_seconds())
DIR_PUBLIC_DATA = os.path.join(WORKSPACE, 'ml-20m/public_data/')


def split_by_timestamp(df, cut_timestamp):
    # chia dataset theo timestamp
    df_public = df[df.timestamp < cut_timestamp]
    df_private = df[df.timestamp >= cut_timestamp]
    return (df_public, df_private)


def prepare_like_problem(df):
    # Chuyển từ rating -> hệ khuyến nghị
    like_theshold = 3.0
    filtered_df = df.loc[df.rating > like_theshold, :]
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df['like'] = 1
    return filtered_df[['userId', 'movieId', 'like', 'timestamp']]


def one_hot_encoding(df, column):
    df = df.copy()
    other_columns = df.columns.tolist()
    other_columns.remove(column)

    # one hot encoding
    one_hot = pd.get_dummies(df[column])
    one_hot.columns = ['{}={}'.format(column, c) for c in one_hot.columns.tolist()]
    df = df.merge(one_hot, right_index=True, left_index=True) \
           .drop([column], axis=1) \
           .groupby(other_columns, as_index=False, sort=False).sum()

    return df


def _segment_title_to_one_hot_encoding(df_movies):
    # text preprocessing for title
    path_stopwords = os.path.join(WORKSPACE, 'stopwords.txt')
    stopwords = [line.strip() for line in open(path_stopwords, 'r').readlines()]
    vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=2)
    tfidf = vectorizer.fit_transform(df_movies['title'])

    map_movie_id = {i: movie_id for i, movie_id in enumerate(df_movies['movieId'].tolist())}

    cx = tfidf.tocoo()
    df = pd.DataFrame(
        {'movieId': [map_movie_id[i] for i in cx.row], 'word_id': cx.col, 'r': cx.data})
    df.loc[:, 'word_id'] = df['word_id'].apply(lambda x: 'title={}'.format(x))

    df = pd.pivot_table(df, index='movieId', columns='word_id', values='r', fill_value=0.0) \
           .reset_index(drop=False)

    df_movies = df_movies.merge(df)
    df_movies = df_movies.drop(['title'], axis=1)
    return df_movies


def refine_movies(df):
    # Tách 'title' thành 2 features : year, title
    df.loc[:, 'year'] = df['title'].str.extract(pat=r'(?P<year>\d{4})').astype('float32')
    df = df.dropna()
    df.loc[:, 'year'] = df['year'].apply(lambda x: int(x))
    df.loc[:, 'title'] = (df['title'].str.extract(r'^(.*) \(....\) *$'))[0]
    df.loc[:, 'genres'] = df['genres'].str.split('|')
    return df


def get_movie_feature(df_movies, df_genome_scores):
    # Merge df -> tạo thêm feature cho đối tượng movie
    # One hot coding - genres
    df_movies = one_hot_encoding(df_movies, 'genres')
    df_movies = df_movies.drop(['genres=(no genres listed)'], axis=1)

    df_movies = _segment_title_to_one_hot_encoding(df_movies)
    #
    df_genome_scores.loc[:, 'tagId'] = \
        df_genome_scores['tagId'].apply(lambda x: 'tagId={}'.format(x))
    df_genome_scores = \
        pd.pivot(df_genome_scores, index='movieId', columns='tagId', values='relevance') \
          .reset_index(drop=False)

    movie_feature = pd.merge(df_movies, df_genome_scores, on='movieId')
    return movie_feature


def main():
    # load data
    df_ratings = pd.read_csv(os.path.join(DIR_MovieLens20M, 'ratings.csv'))
    df_movies = pd.read_csv(os.path.join(DIR_MovieLens20M, 'movies.csv'))
    df_genome_scores = pd.read_csv(os.path.join(DIR_MovieLens20M, 'genome-scores.csv'))

    # create movie feature
    df_movies = refine_movies(df_movies)
    df_movie_feature = get_movie_feature(df_movies, df_genome_scores)

    # split
    df_ratings_public, df_ratings_private = split_by_timestamp(df_ratings, CUT_TIMESTAMP)

    df_movie_feature_public = df_movie_feature[df_movie_feature.year < CUT_YEAR]
    df_movie_feature_private = df_movie_feature[df_movie_feature.year >= CUT_YEAR]

    # create like data
    df_likes_public = prepare_like_problem(df_ratings_public)
    df_likes_private = prepare_like_problem(df_ratings_private)

    # save public data
    df_ratings_public.reset_index(drop=True)\
                     .to_pickle(os.path.join(DIR_PUBLIC_DATA, 'ratings_pub.pkl'))
    df_likes_public.reset_index(drop=True) \
                   .to_pickle(os.path.join(DIR_PUBLIC_DATA, 'likes_pub.pkl'))
    df_movie_feature_public.reset_index(drop=True) \
                           .to_pickle(os.path.join(DIR_PUBLIC_DATA, 'movie_feature_pub.pkl'))

    # save private data
    df_ratings_private.reset_index(drop=True).to_pickle(os.path.join(WORKSPACE, 'ratings_prv.pkl'))
    df_likes_private.reset_index(drop=True).to_pickle(os.path.join(WORKSPACE, 'likes_prv.pkl'))
    df_movie_feature_private.reset_index(drop=True) \
                            .to_pickle(os.path.join(WORKSPACE, 'movie_feature_prv.pkl'))


if __name__ == '__main__':
    main()
