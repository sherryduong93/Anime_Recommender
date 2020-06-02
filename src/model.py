import pandas as pd

def content_based(anime_full):
    anime_map = anime_full[['anime_id','name','title_english']]
    anime_map.set_index('anime_id')
    anime_similarity_cosdf = pd.read_csv('/Users/sherryduong/Documents/Galvanize/Anime_Recommender/models/baseline_contentbased.csv')
    anime_id = input('Find Your Favorite Anime Below, and paste the ID:')
    rec_ids = anime_similarity_cosdf.iloc[int(anime_id),:].sort_values(ascending=False)[:10].index
    print('Based On the anime referenced, you would endjoy:')
    return anime_map[anime_map['anime_id'].isin(rec_ids)].set_index('anime_id')


def find_id(anime_full):
    anime_map = anime_full[['anime_id','name','title_english']]
    anime_map.set_index('anime_id')
    keyword = input('Enter title keywords here to search:')
    return anime_map[anime_map['name'].str.contains(keyword.title())==True]

def popularity_rec(anime_full):
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