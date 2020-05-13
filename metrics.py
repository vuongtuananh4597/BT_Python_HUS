import numpy as np


def recall_k(ratings, model, X_valid, k, threshold):
    """Recall@k metric
    """
    user_list = X_valid.userID.unique().tolist()
    recall_k = []
    for user in user_list:
        tp, fn, fp, tn = 0, 0, 0, 0
        true = ratings[ratings.userID == user].sort_values(by='rating', ascending=False) \
                                              .loc[:, ['movieID', 'rating']][:k]
        false = pd.DataFrame(model.predict_for_user(user).items(), columns=['movieID', 'rating']) \
                                                       .sort_values(by='rating', ascending=False)[:k]

        true.loc[true.rating < threshold, 'rating'] = 0
        true.loc[true.rating >= threshold, 'rating'] = 1
        false.loc[false.rating < threshold, 'rating'] = 0
        false.loc[false.rating >= threshold, 'rating'] = 1
        
        for movieid in df1.iloc[np.where(df1.rating == 1)[0], 0]:
            if any(df2.loc[df2.movieID == movieid, 'rating'] == 1):
                tp += 1
            else:
                fn += 1
        recall = tp / (tp + fn)
        recall_k.append(recall)
    
    return recall_k