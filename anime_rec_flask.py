from flask import Flask, render_template, redirect, url_for, request
from random import random
from src.data_funcs import *
from src.model_funcs import *
app = Flask(__name__)


anime_df, rating_df, anime_meta, users_meta = import_data()
anime_full = full_anime_df(rating_df, anime_df, anime_meta)
anime_map = anime_full[['anime_id','name','title_english', 'type']]
df = anime_map[:10]
simp_df = sim_mat(anime_full, ver='genre')

@app.route('/', methods=["POST","GET"])
def default():
    return redirect(url_for("home"))

@app.route('/home', methods=["POST","GET"])
def home():
    if request.method == "POST":
        media_type1 = request.form["mt"].title()
        keyword1 = request.form["kw"].title()
        return redirect(url_for("search", media_type=media_type1, keyword=keyword1))
    else:
        return render_template("submit.html")

@app.route("/search_results/<media_type>-<keyword>")
def search(media_type, keyword):
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
    return render_template('find_id.html',  search_results=[result.to_html(classes='data')], titles_search=result.columns.values)

@app.route('/engine', methods=["POST","GET"])
def engine():
    if request.method == "POST":
        anime_id2 = request.form["id"]
        return redirect(url_for("recommendations", an_id=anime_id2))
    else:
        return render_template("anime_idform.html")


@app.route('/recommendations/<an_id>', methods=("POST", "GET"))
def recommendations(an_id):
    anime_id = int(an_id)
    media_type = anime_map[anime_map['anime_id']==anime_id]['type'].values
    anime_name = anime_map[anime_map['anime_id']==anime_id]['name'].values
    anime_eng = anime_map[anime_map['anime_id']==anime_id]['title_english'].values
    if media_type == 'Movie':
        type_ids = anime_map[anime_map['type']=='Movie']['anime_id']
        rec_ids = simp_df.loc[anime_id,simp_df.columns.isin(type_ids)].sort_values(ascending=False)[1:11].index
        cb= anime_map[anime_map['anime_id'].isin(rec_ids)].set_index('anime_id')
    elif media_type == 'TV' or media_type == 'OVA' or media_type == 'ONA':
        ova = anime_map['type']=='OVA'
        ona = anime_map['type']=='ONA'
        tv = anime_map['type']=='TV'
        type_ids = anime_map[(ova) | (ona) | (tv)]['anime_id']
        rec_ids = simp_df.loc[anime_id,simp_df.columns.isin(type_ids)].sort_values(ascending=False)[1:11].index
        cb= anime_map[anime_map['anime_id'].isin(rec_ids)].set_index('anime_id')
    else:
        rec_ids = simp_df.loc[anime_id,:].sort_values(ascending=False)[1:11].index
        cb= anime_map[anime_map['anime_id'].isin(rec_ids)]
    return render_template('simple.html',  content_based=[cb.to_html(classes='data')], titles_CB=cb.columns.values, 
    collab_filt=[df.to_html(classes='data')], titles_CF=df.columns.values, anime_name = str(anime_name)[2:-2], media_type = str(media_type)[2:-2], anime_eng = str(anime_eng)[2:-2])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8089, debug=True)
