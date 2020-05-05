import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

PATH = "/Users/vietnd/Documents/Datasets/ml-20m"


# lấy cỡ của folder data
def get_shape():
    for root, _, file in os.walk(PATH):
        for i in range(len(file)):
            if file[i].endswith('.csv'):
                data = pd.read_csv(os.path.join(PATH, file[i]))
                print("{} : {}".format(file[i], data.shape))


# Loading data
ratings = pd.read_csv(os.path.join(PATH, 'ratings.csv'))
movies = pd.read_csv(os.path.join(PATH, 'movies.csv'))
tags = pd.read_csv(os.path.join(PATH, 'tags.csv'))
genome_scores = pd.read_csv(os.path.join(PATH, 'genome-scores.csv'))
genome_tags = pd.read_csv(os.path.join(PATH, 'genome-tags.csv'))


# 1.ratings.csv
def analysis_rating():
    user_count = len(ratings.userId.unique())
    rated_movie = len(ratings.movieId.unique())
    all_movie = len(movies.movieId.unique())
    print("Số lượng user đã đánh giá item: ", user_count)
    print("Số lượng item được đánh gía: ", rated_movie)
    print("Số lượng toàn bộ item có: ", all_movie)
    print("Số lượng item không được đánh giá: ", all_movie - rated_movie)
    print('Trung bình 1 user đánh giá cho {:.04f} itmes'.format(ratings.shape[0] / user_count))
    print("Tỷ lệ số lượng giá trị ratings có / Tổng giá trị phải có :", ratings.shape[0]*100/(all_movie*user_count))

    fig, (ax1, ax2) = plt.subplots(2)
    fig.tight_layout()
    # Rating's value distribution
    ax1.hist(ratings.rating, bins=[i for i in np.arange(0.5, 5.5, 0.5)])
    ax1.set_title('Rating distribution')
    # Timestamp distribution
    ratings['time'] = pd.to_datetime(ratings.timestamp, unit='s')
    ax2.hist(ratings['time'], bins=20)
    ax2.set_title('Timestamp distribution')
    plt.show()


# 2.movies.csv
# movieId, title, genres
def analysis_movie():
    movies['year'] = movies.title.str.extract(pat=r'(?P<year>\d{4})').dropna()
    # Check nan
    print(movies.loc[movies.year.isnull(), ['title', 'year']])
    # Year distribution
    plt.hist(movies.year)
    # Count number of genre
    genres = dict()
    for _, value in movies.genres.iteritems():
        for genre in value.split('|'):
            genres[genre] = genres.get(genre, 0) + 1
    plt.show()


# genome-scores.csv (movieId, tagId, re)  : tương quan giữa movie
# tags.csv (userId, movieId, tag, timestamp)
# genome-tags.csv (tagId, tag)
def analysis_tag():
    no_movie = len(genome_scores.movieId.unique())
    print("Tỷ lệ item được tag: {}/{} ".format(no_movie, len(movies.movieId.unique())))
    print("Trung bình mỗi item được gán {} thẻ".format(genome_scores.shape[0] / no_movie))
    # Check
    # print(np.unique(genome_scores.movieId.value_counts().values))
    tmp = genome_scores.loc[genome_scores.movieId == 1, 'relevance'].values
    plt.hist(tmp)
    plt.title("Relevance's score distribution of 1st item")
    plt.show()
