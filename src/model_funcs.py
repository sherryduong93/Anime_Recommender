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

def find_id(anime_full, keyword, media_type):
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
    # media_type = input('Please select Movie, TV, or Both:').title()
    # keyword = input('Enter title keywords here to search:').title()
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
        rec_ids = simp_df.loc[anime_id,simp_df.columns.isin(type_ids)].sort_values(ascending=False)[1:11].index
        print('Based On the anime referenced, you would enjoy:')
        return anime_map[anime_map['anime_id'].isin(rec_ids)].set_index('anime_id')
    elif media_type == 'TV' or media_type == 'OVA' or media_type == 'ONA':
        ova = anime_map['type']=='OVA'
        ona = anime_map['type']=='ONA'
        tv = anime_map['type']=='TV'
        type_ids = anime_map[(ova) | (ona) | (tv)]['anime_id']
        rec_ids = simp_df.loc[anime_id,simp_df.columns.isin(type_ids)].sort_values(ascending=False)[1:11].index
        print('Based On the anime referenced, you would enjoy:')
        return anime_map[anime_map['anime_id'].isin(rec_ids)].set_index('anime_id')
    else:
        rec_ids = simp_df.loc[anime_id,:].sort_values(ascending=False)[1:11].index
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