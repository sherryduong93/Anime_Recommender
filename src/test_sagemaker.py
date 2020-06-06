import boto3
import pandas as pd
from sagemaker import get_execution_role
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error

role = 'AmazonSageMaker-ExecutionRole-20200524T114773'

anime_df = pd.read_csv('s3://animerec/Anime_Recommender/data/anime.csv')
rating_df = pd.read_csv('s3://animerec/Anime_Recommender/data/rating.csv')
#Remove the -1's, which are no values for the ratings
rating_df = rating_df[rating_df['rating']!=-1]
anime_meta = pd.read_csv('s3://animerec/Anime_Recommender/data/AnimeList_Meta.csv')
users_meta = pd.read_csv('s3://animerec/Anime_Recommender/data/UserList_Meta.csv')

def weighted_rating(x,rating_count_col, avg_rating_col):
    m = x[rating_count_col].quantile(0.80)
    C = x[avg_rating_col].mean()
    v = x[rating_count_col]
    R = x[avg_rating_col]
    # Compute the weighted score
    return (v/(v+m) * R) + (m/(m+v) * C)


def full_anime_df(rating_df, anime_df, anime_meta):
    #Get the total number of ratings per anime
    count_ratings = rating_df.groupby('anime_id').count().rename(columns={'rating': 'num_ratings'})['num_ratings']
    
    #Combine the meta data with the anime data, and rating data
    anime_full = anime_df.merge(right=anime_meta, how='left', on='anime_id')
    anime_full = anime_full.merge(right=count_ratings, how='left', on='anime_id')
    anime_full = anime_full.drop(columns=['title','title_japanese','title_synonyms', 'type_x',
                                      'episodes_y', 'airing', 'score','scored_by', 'members_y', 'background',
                                     'licensor', 'premiered', 'broadcast', 'related', 'genre_x', 'aired_string'])
    anime_full = anime_full.rename(columns={'rating_x': 'avg_rating','rating_y': 'rating_type', 'genre_y':'genre', 
                                        'members_x': 'members', 'episodes_x':'episodes', 'type_y':'type', 0: 'weighted_rating'})
    anime_full = pd.concat([anime_full, weighted_rating(anime_full, 'members','avg_rating')], axis=1)
    anime_full = anime_full.rename(columns={0: 'weighted_rating'})
    
    #Shortening the rating type categories
    rating_type_dict = {'PG-13 - Teens 13 or older': 'PG-13', 'R - 17+ (violence & profanity)': 'R',
                   'PG - Children': 'PG', 'G - All Ages': 'G', 'R+ - Mild Nudity': 'R+', 
                   'Rx - Hentai':'RX', 'None': 'Unknown'}
    anime_full['rating_type'] = anime_full['rating_type'].map(rating_type_dict).fillna('Unknown')
    
    #Filling NaNs
    anime_full['genre'] = anime_full['genre'].fillna('Unknown')
    anime_full['studio'] = anime_full['studio'].fillna('Unknown')
    anime_full['producer'] = anime_full['producer'].fillna('Unknown')
    
    #Formatting the anime titles
    anime_full['name'] = anime_full['name'].str.title()
    anime_full['title_english'] = anime_full['title_english'].str.title()
    return anime_full

def explode_text(anime_full):
    exp_anime = anime_full.copy()
    #explode all columns
    exp_anime['genre'] = exp_anime['genre'].transform(lambda x: x.split(','))
    exp_anime = exp_anime.explode('genre')
    exp_anime['genre'] = exp_anime['genre'].transform(lambda x: x[1:] if x[0]==" " else x)
    exp_anime['studio'] = exp_anime['studio'].transform(lambda x: x.split(','))
    exp_anime = exp_anime.explode('studio')
    exp_anime['producer'] = exp_anime['producer'].transform(lambda x: x.split(','))
    exp_anime = exp_anime.explode('producer')
    exp_anime['studio'] = exp_anime['studio'].transform(lambda x: x[1:] if x[0]==' ' else x)
    exp_anime['producer'] = exp_anime['producer'].transform(lambda x: x[1:] if x[0]==' ' else x)
    return exp_anime

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances


def sim_mat(anime_full, ver='basic'):
    '''
    Returns similarity matrix with cosine similarity based on the anime_dataframe provided

    INPUT - 
    anime_full: Formatted anime dataframe using the full_anime_df function, size (nxm)
    ver: version of similarity matrix available, select from:
        basic: 
        genre: The above + Genre based similarity
        adv: genre + popular studio & producer 
    
    OUTPUT -
    dataframe containing the similaries between each anime in anime_full, size (nxn)
    '''
    if ver=='basic':
        basic_rec = anime_full[['anime_id','type','source','rating_type']] #removed 'weighted_rating' to test
        basic_rec = pd.get_dummies(basic_rec, columns=['type','source','rating_type']).set_index('anime_id')
        # basic_rec = basic_rec.dropna(axis=0, subset=['weighted_rating']) 
        anime_similarity_cos = cosine_similarity(basic_rec)
        anime_similarity_cosdf = pd.DataFrame(anime_similarity_cos, index=basic_rec.T.columns, columns=basic_rec.T.columns)
        return anime_similarity_cosdf
    elif ver=='genre':
        ContBased_2 = anime_full[['anime_id','type','source','rating_type','weighted_rating']]
        vect = CountVectorizer()
        count_vect = vect.fit_transform(anime_full['genre'])
        features = vect.get_feature_names()
        genre_matrix = count_vect.toarray()
        genre_df = pd.DataFrame(genre_matrix, columns =features, index=anime_full['anime_id'])
        ContBased_2 = ContBased_2.merge(right=genre_df,how='inner',on='anime_id')
        ContBased_2 = pd.get_dummies(ContBased_2, columns=['type','source','rating_type']).set_index('anime_id')
        ContBased_2 = ContBased_2.dropna(axis=0, subset=['weighted_rating'])
        df = ContBased_2
        anime_similarity_cos = cosine_similarity(df)
        anime_similarity_cosdf = pd.DataFrame(anime_similarity_cos, index=df.T.columns, columns=df.T.columns)
        return anime_similarity_cosdf
    elif ver=='adv':
        ContBased_3 = anime_full[['anime_id','type','source','rating_type','weighted_rating','studio','producer']]
        vect = CountVectorizer()
        count_vect = vect.fit_transform(anime_full['genre'])
        features = vect.get_feature_names()
        genre_matrix = count_vect.toarray()
        genre_df = pd.DataFrame(genre_matrix, columns =features, index=anime_full['anime_id'])
        top_studios = ['Studio Chizu', 'Marvy Jack', 'Bandai Namco Pictures', 'White Fox', 'Purple Cow Studio Japan', 
        'Shuka', 'Egg Firm', 'Square Enix', 'Wit Studio', 'Lay-duce', 'Studio Ghibli', 'Graphinica', 'David Production', 
        'Bridge', 'Animation Do', 'P.A. Works', 'Kyoto Animation', 'Manglobe', 'Artland', 'Hoods Drifters Studio']
        top_producers = ['Studio Moriken', 'Quaras', 'Seikaisha', 'Mad Box', 'Forecast Communications', 'StudioRF Inc.', 
        'CIC', 'TAP', 'Miracle Robo', 'Madoka Partners', 'Animation Do', 'Studio Wombat', 'GYAO!', 
        'Shingeki no Kyojin Team', 'C &amp; I entertainment', 'Top-Insight International Co.', 'LTD.', 
        'East Japan Marketing &amp; Communications', 'Audio Highs','Banpresto']
        for studio in top_studios:
            ContBased_3[studio] = ContBased_3['studio'].transform(lambda x: 1 if studio in x else 0)
        for producer in top_producers:
            ContBased_3[producer] = ContBased_3['producer'].transform(lambda x: 1 if studio in x else 0)
        ContBased_3 = ContBased_3.merge(right=genre_df,how='inner',on='anime_id')
        ContBased_3 = pd.get_dummies(ContBased_3, columns=['type','source','rating_type']).set_index('anime_id')
        ContBased_3 = ContBased_3.drop(columns=['studio','producer'])
        ContBased_3 = ContBased_3.dropna(axis=0, subset=['weighted_rating'])
        df = ContBased_3.copy()
        anime_similarity_cos = cosine_similarity(df)
        anime_similarity_cosdf = pd.DataFrame(anime_similarity_cos, index=df.T.columns, columns=df.T.columns)
        return anime_similarity_cosdf
    else:
        return 'Please Select basic, genre, or adv for ver'

def find_id(anime_full):
    '''
    Helper function to help the user find the anime_id for a specific anime using
    keyword search.

    INPUT - 
    anime_full: Formatted anime dataframe using the full_anime_df function

    OUTPUT -
    dataframe containing the anime_id, anime_title(Japanese), anime_title(English), 
    and media_type of results matching the keyword search
    '''
    anime_map = anime_full[['anime_id','name','title_english', 'type']]
    media_type = input('Please select Movie, TV, or Both:').title()
    keyword = input('Enter title keywords here to search:').title()
    keyword_search_name = anime_map['name'].str.contains(keyword)==True
    keyword_search_eng = anime_map['title_english'].str.contains(keyword)==True
    if media_type == 'Both':
        result = anime_map[((keyword_search_name) | (keyword_search_eng))]
    elif media_type == 'Movie':
        result = anime_map[((keyword_search_name) | (keyword_search_eng)) & (anime_map['type']=='Movie')]
    elif media_type == 'Tv':
        ova = anime_map['type']=='OVA'
        ona = anime_map['type']=='ONA'
        tv = anime_map['type']=='TV'
        result = anime_map[((keyword_search_name) | (keyword_search_eng)) & ((ova) | (ona) | (tv))]
    else:
        result = 'Incorrect media type, please select from Movie, TV, or Both'
    return result

def content_based(anime_full, simp_df):
    '''
    Content based recommender system
    
    INPUT:
    anime_full: Formatted anime dataframe using the full_anime_df function
    simp_df: Similarity matrix for all anime in anime_df

    OUTPUT: 
    dataframe: Dataframe containing the top 10 recommended anime according to the
    similarities between animes. Dataframe will contain anime_id, anime_title(Japanese), 
    anime_title(English), and media_type
    '''
    anime_map = anime_full[['anime_id','name','title_english', 'type']]
    anime_id = int(input('Find Your Favorite Anime Below, and paste the ID:'))
    media_type = anime_map[anime_map['anime_id']==anime_id]['type'].values
    anime_name = anime_map[anime_map['anime_id']==anime_id]['name'].values
    print(f'{anime_name}: {media_type}')
    if media_type == 'Movie':
        type_ids = anime_map[anime_map['type']=='Movie']['anime_id']
        rec_ids = rec_ids = simp_df.iloc[simp_df.index==anime_id,simp_df.columns.isin(type_ids)].T.sort_values(by=anime_id, ascending=False)[1:11].index
        print('Based On the anime referenced, you would enjoy:')
        return anime_map[anime_map['anime_id'].isin(rec_ids)].set_index('anime_id')
    elif media_type == 'TV' or media_type == 'OVA' or media_type == 'ONA':
        ova = anime_map['type']=='OVA'
        ona = anime_map['type']=='ONA'
        tv = anime_map['type']=='TV'
        type_ids = anime_map[(ova) | (ona) | (tv)]['anime_id']
        rec_ids = simp_df.iloc[simp_df.index==anime_id,simp_df.columns.isin(type_ids)].T.sort_values(by=anime_id, ascending=False)[1:11].index
        print('Based On the anime referenced, you would enjoy:')
        return anime_map[anime_map['anime_id'].isin(rec_ids)].set_index('anime_id')
    else:
        rec_ids = simp_df.iloc[simp_df.index==anime_id,:].T.sort_values(by=anime_id, ascending=False)[1:11].index
        print('Based On the anime referenced, you would enjoy:')
        return anime_map[anime_map['anime_id'].isin(rec_ids)].set_index('anime_id')[1:11]

def popularity_rec(anime_full):
    '''
    FYI: This is incomplete and not yet running
    '''
    exp_anime = anime_full.copy()
    exp_anime['genre'] = exp_anime['genre'].transform(lambda x: x.split(','))
    exp_anime = anime_full.explode('genre')
    
    rating_type = input('Please select rating for viewing from the following: All, PG, PG-13, R, R+, Rx:').upper()
    media_type = input('Please select Movie, TV, or Both:').title()
    
    if rating_type == 'ALL':
        genre_list = np.sort(exp_anime['genre'].unique())
        genre_df = exp_anime.copy()
        df = anime_full.copy()
    else:
        genre_list = np.sort(exp_anime[exp_anime['rating_type']==rating_type]['genre'].unique())
        genre_ids = exp_anime[exp_anime['rating_type']==rating_type]['anime_id'].unique()
        genre_df = exp_anime[exp_anime['rating_type']==rating_type]
        df = anime_full.copy()
        df = df[df['anime_id'].isin(genre_ids)]
    
    genre_type = input(f'Please select genre from the following: All or  {genre_list}:').title()
    if media_type == 'Both':
        df = df
    else:
        df = df[df['type']==media_type]
    
    if genre_type == 'All':
        result = df[['anime_id','name','title_english','weighted_rating']].sort_values('weighted_rating',ascending=False)[:10]
    else:
        id_df = genre_df[genre_df['genre']==genre_type].sort_values('weighted_rating', ascending=False)
        top_10 = id_df['anime_id'].unique()[:10]
        result = df[df['anime_id'].isin(top_10)][['anime_id','name','title_english','type','weighted_rating']].sort_values('weighted_rating',ascending=False)
    if result.empty:
        return 'No match - Please try a different combination!'
    else:
        return result

def pred_user_rating(rating_df, sim_mat, user_id, anime_id):
    '''
    INPUT:
    rating_df: matrix with ratings for each anime and user
    sim_mat: similarity matrix with similarity scores for all anime
    utility_may: users(row) and anime_id(columns) with ratings for each anime provided by the user
    user_id: User id of user to predict rating for
    anime_id: Anime_id for anime to predict user rating

    OUTPUT:
    pred_rating: Predicted rating from the user(user_id) for anime(anime_id)
    '''
    anime_ids = rating_df[rating_df['user_id']==user_id]['anime_id'].values
    anime_ids2 = anime_ids[anime_ids!=anime_id]
    final_ids = anime_ids2[np.isin(anime_ids2, sim_mat.columns)]
    ratings = rating_df[(rating_df['user_id']==user_id) & (rating_df['anime_id'].isin(final_ids))]['rating'].values
    sim_mat_red = sim_mat.iloc[:, sim_mat.columns.isin(final_ids)]
    sims = sim_mat_red.iloc[sim_mat_red.index==anime_id, :].values
    pred_rating = np.sum(ratings*sims)/np.sum(sims)
    return pred_rating

    anime_full = full_anime_df(rating_df, anime_df, anime_meta)

sim_mat_basic = sim_mat(anime_full, ver='basic')
sim_mat_basic = sim_mat(anime_full, ver='genre')
entire_pivot = pd.pivot_table(data=rating_df, values='rating', index='user_id', columns='anime_id', fill_value=0)


filt = rating_df.groupby('user_id').count()['rating']
user_ids = filt[filt>50].reset_index()['user_id'].values
over_df = rating_df[rating_df['user_id'].isin(user_ids)]
remaining_df = rating_df[~rating_df['user_id'].isin(user_ids)]
over_df.groupby('user_id').count()['rating'].sort_values()
y=over_df['user_id']
X=over_df.drop(columns=['user_id'])
anime_train, anime_test, user_train, user_test = train_test_split(X, y, test_size = 0.20, random_state = 0, stratify=y)
train_over_split = pd.concat([anime_train, user_train],axis=1)
train = pd.concat([train_over_split, remaining_df], axis=0)
test_df = pd.concat([anime_test, user_test],axis=1)[:100]

test_df['pred_basic'] = test_df.apply(lambda row: pred_user_rating(rating_df, sim_mat_basic, row['user_id'], row['anime_id']), axis=1)
test_df['pred_genre'] = test_df.apply(lambda row: pred_user_rating(rating_df, sim_mat_genre, row['user_id'], row['anime_id']), axis=1)

rmse_basic = np.sqrt(mean_squared_error(test_df['rating'],test_df['pred_basic']))
rmse_genre = np.sqrt(mean_squared_error(test_df['rating'],test_df['pred_genre']))
print(rmse_basic, rmse_genre)