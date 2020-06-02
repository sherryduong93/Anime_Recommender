import pandas as pd

def content_based(anime_full, simp_df):
    anime_map = anime_full[['anime_id','name','title_english', 'type']]
    anime_id = input('Find Your Favorite Anime Below, and paste the ID:')
    media_type = anime_map[anime_map['anime_id']==int(anime_id)]['type'].values
    if media_type == 'Movie':
        type_ids = anime_map[anime_map['type']=='Movie']['anime_id']
        rec_ids = simp_df.iloc[int(anime_id),simp_df.columns.isin(type_ids)].sort_values(ascending=False)[:10].index
        print('Based On the anime referenced, you would endjoy:')
        return anime_map[anime_map['anime_id'].isin(rec_ids)].set_index('anime_id')
    elif media_type == 'Tv' or media_type == 'OVA' or media_type == 'ONA':
        ova = anime_map['type']=='OVA'
        ona = anime_map['type']=='ONA'
        tv = anime_map['type']=='TV'
        type_ids = anime_map[(ova) | (ona) | (tv)]['anime_id']
        rec_ids = simp_df.iloc[int(anime_id),simp_df.columns.isin(type_ids)].sort_values(ascending=False)[:10].index
        print('Based On the anime referenced, you would endjoy:')
        return anime_map[anime_map['anime_id'].isin(rec_ids)].set_index('anime_id')
    else:
        rec_ids = simp_df.iloc[int(anime_id),:].sort_values(ascending=False)[:10].index
        print('Based On the anime referenced, you would endjoy:')
        return anime_map[anime_map['anime_id'].isin(rec_ids)].set_index('anime_id')


def find_id(anime_full):
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

def popularity_rec(anime_full):
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