from flask import Flask, render_template, redirect, url_for, request
from random import random
from src.data_funcs import *
from src.model_funcs import *
app = Flask(__name__)


anime_df, rating_df, anime_meta, users_meta = import_data()
anime_full = full_anime_df(rating_df, anime_df, anime_meta)
anime_map = anime_full[['anime_id','name','title_english', 'type']]
simp_df = sim_mat(anime_full, ver='genre')

otherusers_df = pd.read_csv('model/otherusers_rec2.csv').drop(columns=['Unnamed: 0']) #Use rec2 for 50 users, rec for 10
yourrecs_df = pd.read_csv('model/your_recs2.csv').drop(columns=['Unnamed: 0']) #Use rec2 for 50 users, rec for 10

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
    result = find_id(anime_map, keyword, media_type)
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
    anime_name, media_type, anime_eng, cb = content_based(anime_id, anime_map, simp_df)
    collab_filt = other_users(anime_id, otherusers_df,yourrecs_df, anime_map)
    return render_template('simple.html',  content_based=[cb.to_html(classes='data')], titles_CB=cb.columns.values, 
    collab_filt=[collab_filt.to_html(classes='data')], titles_CF=collab_filt.columns.values, anime_name = str(anime_name)[2:-2], media_type = str(media_type)[2:-2], anime_eng = str(anime_eng)[2:-2])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8089, debug=True)
