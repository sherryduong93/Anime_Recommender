import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from model_funcs import *
from data_funcs import *

def fillna_ratingspivot(ratings_pivot, method='item_avg'):
    if method=='zero':
        return ratings_pivot.fillna(0)
    elif method=='user_avg':
        return ratings_pivot.fillna(ratings_pivot.mean(axis=0)) 
    elif method=='item_avg':
        return ratings_pivot.T.fillna(ratings_pivot.mean(axis=1)).T
    else:
        return 'Please select method from: zero, user_avg, or item_avg'

def rating_pivot(anime_full, rating_df, method='item_avg'):
    anime_map = anime_full[['anime_id','name','title_english', 'type']]
    user_grp = rating_df.groupby('user_id').count()['rating']
    item_grp = rating_df.groupby('anime_id').count()['rating']
    
    users_over_thresh = user_grp[user_grp>300].index
    anime_over_thresh = item_grp[item_grp>2500].index

    #Only take the users & anime over the thresholds
    reduced_df = rating_df[(rating_df['user_id'].isin(users_over_thresh)) & (rating_df['anime_id'].isin(anime_over_thresh))]

    ratings_pivot = pd.pivot_table(data=reduced_df, values='rating', index='anime_id', columns='user_id')
    ratings_pivot1 = fillna_ratingspivot(ratings_pivot, method=method)
    return ratings_pivot1


def knn_rec(anime_full, rating_df, anime_id, method='item_avg'):
    anime_map = anime_full[['anime_id','name','title_english', 'type']]
    ratings_pivot = rating_pivot(anime_full, rating_df, method=method)
    ratings_csr = csr_matrix(ratings_pivot)
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(ratings_csr)
    distances, indices = model_knn.kneighbors(ratings_pivot.iloc[ratings_pivot.index==anime_id,:], n_neighbors = 11)
    for i in range(0, len(distances.flatten())):
        if i == 0:
            anime_name = anime_map[anime_map['anime_id']==anime_id]['name'].values
            print(f'Recommendations for {anime_id} {anime_name}:\n')
        else:
            idx = ratings_pivot.index[indices.flatten()[i]]
            print('{0}: {1}, with distance of {2}:'.format(i, anime_map[anime_map['anime_id']==idx]['name'].values, distances.flatten()[i]))

def simple_svd_rec(anime_full, rating_df, method='item_avg'):
    ratings = rating_pivot(anime_full, rating_df, method=method)
    SVD = TruncatedSVD(n_components=10, random_state=0)
    matrix = SVD.fit_transform(ratings)
    corr_mat = np.corrcoef(matrix)
    corr_df = pd.DataFrame(corr_mat, index=ratings.index, columns=ratings.index)
    return content_based(anime_full, corr_df)