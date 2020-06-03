from model_funcs import *
from data_funcs import *

anime_df, rating_df, anime_meta, users_meta = import_data()
anime_full = full_anime_df(rating_df, anime_df, anime_meta)

simi_mat = sim_mat(anime_full, ver='genre')

find_id(anime_full)

content_based(anime_full, simi_mat)