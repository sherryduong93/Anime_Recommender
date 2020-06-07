from flask import Flask, render_template, redirect, url_for, request
from random import random
from src.data_funcs import *
from src.model_funcs import *
app = Flask(__name__)


anime_df, rating_df, anime_meta, users_meta = import_data()
anime_full = full_anime_df(rating_df, anime_df, anime_meta)
# anime_full.loc[anime_id, 'image_url'] = '<img src="https://cdn.myanimelist.net/images/anime/10/73274.jpg" alt="flowers" style="width:100px;height:150px;">'
anime_map = anime_full[['anime_id','name','title_english', 'type', 'image_url']]
simp_df = sim_mat(anime_full, ver='genre')
most_popular_ids = anime_full.sort_values('weighted_rating', ascending=False)['anime_id'].values
most_popular_ids = most_popular_ids[most_popular_ids != 918][:10]


otherusers_df = pd.read_csv('model/otherusers_rec2.csv').drop(columns=['Unnamed: 0']) #Use rec2 for 50 users, rec for 10
yourrecs_df = pd.read_csv('model/your_recs2.csv').drop(columns=['Unnamed: 0']) #Use rec2 for 50 users, rec for 10

@app.route('/home', methods=["POST","GET"])
@app.route('/', methods=["POST","GET"])
def home():
    if request.method == "POST":
        media_type1 = request.form["mt"].title()
        keyword1 = request.form["kw"].title()
        if media_type1 == "" or keyword1 == "":
            return redirect(url_for("home", content="Please fill out the form."))
        return redirect(url_for("search", media_type=media_type1, keyword=keyword1))
    else:
        return render_template("submit.html")

@app.route("/search_results/<media_type>-<keyword>")
def search(media_type, keyword):
    result = find_id(anime_map, keyword, media_type)
    return render_template('find_id.html',  search_results=[result.to_html(escape=False, classes='table text-center table-hover table-light table-small w-auto' )], titles_search=result.columns.values)

@app.route('/engine', methods=["POST","GET"])
def engine():
    most_pop = anime_map[anime_map['anime_id'].isin(most_popular_ids)][['anime_id','name','title_english','image_url']]
    most_pop_df = pd.DataFrame([most_pop['image_url'],most_pop['name'], most_pop['title_english']])
    new = most_pop['anime_id'].unique()
    old = most_pop_df.columns
    d = {old[i] : new[i] for i in range(0,10)}
    most_pop_df = most_pop_df.rename(columns=d)
    
    if request.method == "POST":
        anime_id2 = request.form["id"]
        return redirect(url_for("recommendations", an_id=anime_id2))
    else:
        return render_template("anime_idform.html", most_pop_table=[most_pop_df.to_html(escape=False, classes='table-responsive text-center table-hover table-light table-small w-auto' )])


@app.route('/recommendations/<an_id>', methods=("POST", "GET"))
def recommendations(an_id):
    anime_id = int(an_id)
    anime_name, media_type, anime_eng, cb = content_based(anime_id, anime_map, simp_df)
    collab_filt = other_users(anime_id, otherusers_df,yourrecs_df, anime_map)
    return render_template('simple.html',  content_based=[cb.to_html(escape=False, classes='table text-center table-hover table-light table-small w-auto' )], titles_CB=cb.columns.values,
    collab_filt=[collab_filt.to_html(escape=False, classes='table text-center table-hover table-light table-small w-auto')], titles_CF=collab_filt.columns.values, anime_name = str(anime_name)[2:-2], media_type = str(media_type)[2:-2], anime_eng = str(anime_eng)[2:-2])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8089, debug=True)
