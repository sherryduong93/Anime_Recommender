import pandas as pd
import numpy as np

def import_data():
    anime_df = pd.read_csv('data/anime.csv')
    rating_df = pd.read_csv('data/rating.csv')
    #Remove the -1's, which are no values for the ratings
    rating_df = rating_df[rating_df['rating']!=-1]
    anime_meta = pd.read_csv('data/AnimeList_meta.csv')
    users_meta = pd.read_csv('data/UserList_Meta.csv')
    return anime_df, rating_df, anime_meta, users_meta


def weighted_rating(df,rating_count_col, avg_rating_col):
    m = df[rating_count_col].quantile(0.80)
    C = df[avg_rating_col].mean()
    v = df[rating_count_col]
    R = df[avg_rating_col]
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

