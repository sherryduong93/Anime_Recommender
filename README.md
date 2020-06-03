# Find Your Next Anime
![image](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/a4637adf-84a6-4eba-8020-71e9a0452e82/dd99tk0-16be0db0-a59c-45a8-bfbb-252cab02956d.png)

## Motivations & Goals
Since shelter-in-place was enacted, more people have been staying home looking for more ways to pass the time. Like many, I found myself wanting to escape to a world of fantasy, and found that anime was the best way to do this.
<br>However, when I finished one anime, I was surpised how difficult it was to find a similar anime to the one I enjoyed. It required searching on Google and going through various forums to find suggestions that seemed aligned to my tastes.
<br>I have two goals for this project:
<br>1) A content based recommender that will recommend shows/movies similar to my last favorite anime
<br>2) An item-item collaborative filter recommender system that will recommend anime based on the ratings and reviews of users similar to me.

## The Data:
<br>**-anime_df**: 12,294 animes with name, genre, type, number of episodes, avg_rating, and members
<br>**-rating_df**: 7M reviews of 11,200 animes from 73,515 users
<br>**-anime_meta**: 14,478 animes with additional features: English title, dates aired, duration of anime, rating (PG,G,R, etc.), producer, studio, opening & ending theme songs
<br>**-users_meta**: 302,673 unique users with number of episodes watched, along with gender, birthdate, location, membership_date
<br>**-Optional: concepts_titlesonly**: 391706 anime titles & concepts, but no anime_id, so will be difficult to group with current dataset

## Data Cleaning
<br>Ratings: Removed all ratings with '-1' which indicates no rating.
<br>NOTE: Ratings matrix does not have ALL of the ratings provided by the users that make up the average rating column in the anime_df, will need to consider this for the collaborative filter based recommender. 
<br>**Combining dataframes to get one large dataframe with all metadata for each anime**
<br>The function for below cleaning is stored in src/data_funcs.py, full_anime_df() function: 
<br>The anime data and rating data were combined in order to calculated weighted rating based on number of members that rated the anime.
<br>The anime meta data was joined with the anime dataset through the anime_id.
<br>The rating types were streamlined to: G, PG, PG-13, R, R+,Rx, and "Unknown" for any blank values.
<br>The blanks for genre, studio, and producer were also filled with "Unknown".
<br>Dropped any duplicate columns contained in both the anime_meta dataset, and the anime dataset. Dropped any irrelevant columns with many nulls, or data that is not useful.
<br>**Exploding the Genre, Producer, & Studio columns to see trends**
<br>The function for performing this operation is the explode_text() function in src/data.py.

## EDA
**Comparing the average rating to the weighted ratings**
-avg_rating from the anime dataset: 6.473902
-weighted_rating average from the anime dataset: 6.654531
![image](images/ratings_dist.png)
-Weighted ratings are more closely centered around the mean, right more positive right skew, meanwhile average ratings are more normally distributed.
-Average ratings from the user rating dataset: 7.8
-Could be due to rating data missing some ratings of animes listed in the anime dataset

**Ratings across different features**
-No major differences between anime type or source
![image](images/ratings_per_type.png)
![image](images/ratings_per_source.png)
<br>-On average (not weighted), the PG-13 & R animes are doing slightly better than other rating types.
![image](images/ratings_per_rating_type.png)
-Comparing different genres, there are some genres that do much better/worse than majority.
![image](images/ratings_per_genre.png)
-Comparing the top 20 studios and producers, there are clear studios that are more highly rated by users.
<br><img src="images/ratings_per_studio.png" width="425"/> <img src="images/ratings_per_producers.png" width="425"/> 


## Content Based Recommender System:
**Anime_id Keyword**
<br>-To help users search for the anime_id desired, a helper function called find_id() in src/model_funcs.py was created, which will return all titles that have the keyword.
<br>**Baseline Content Based Recommender:**
<br>-Features: Type (Movie, TV, etc.), Source (Manga, Music, Book, etc.), Rating Type (PG, R, etc.), and Weighted Rated.
<br>-Similarity Metrics: Tested Cosine Similarity & Pairwise Distance. Spot-checking a few popular animes in each genre, Pairwise Distance performs the best with genre related recommendations, but is recommending unpopular anime, cosine similarity is recommending the popular anime, but not good at narrowing down to the right genre. 
<br>Results: (With cosine similarity), looking up recommendations for "Inuyasha"
<br><img src="images/Content_Based_Compare_TV1.png" width="425"/> <img src="images/Content_Based_Movie_1.png" width="425"/> 
<br>**Content Based Recommender Iteration 2:**
<br>-Added dummified genre to the content based model
<br>-Based on the EDA, some producers/studios have higher ratings overall than others, only the top 20 studios and producers will be captured in the next content based model.
<br>-Overall, the genre significantly helped with the recommendations. The recommender is now recommending more highly rated anime that is closer to the genre specified, though not perfect
<br><img src="images/Content_Based_Compare_TV2.png" width="425"/> <img src="images/Content_Based_Movie2.png" width="425"/> 
<br>**Content Based Recommender Iteration 3**
<br>-Adding Studio/Producers had almost no impact on the similarity matrix of the iteration with only genre added.
<br>-Explored clusters of producers & studios, but as there are many duplicates in multiple clusters, did not think this would be worth exploring.
<p>Clusters of Producers:
<br>0, Bandai Visual, Pink Pineapple, Lantis, Sanrio, Fuji TV
<br>1, Unknown, Bandai Visual, Aniplex, NHK, TV Tokyo
<br>2, TV Tokyo, Tokyo Movie Shinsha, Sanrio, Sotsu, Milky Animation Label
<br>3, NHK, Sanrio, Tokyo Movie Shinsha, Fuji TV, Milky Animation Label
<br>4, Aniplex, Tokyo Movie Shinsha, Sanrio, Fuji TV, Milky Animation Label</p>
<p>Clusters of Studios:
<br>0, Sunrise, Madhouse, Production I.G, Studio Pierrot, TMS Entertainment
<br>1, Unknown, Sunrise, Madhouse, J.C.Staff, Studio Pierrot
<br>2, Studio Deen, Toei Animation, Sunrise, OLM, Xebec
<br>3, J.C.Staff, Toei Animation, Sunrise, OLM, Xebec
<br>4, Toei Animation, Unknown, Nippon Animation, OLM, Tatsunoko Production</p>

## Simple Collaborative Filter Recommenders with Correlation, KNN & SVD
**Rating Data Statistics:**
<br>On average, each user provides 90 ratings, median number of ratings given per user is 45
<br>On average, each anime has 638 ratings, median number of ratings provided per anime is 57
![image](images/ratings_dist.png)
<br>For our simple collaborative filter recommenders, we want to recommend the most popular movies from our most active users. I will be removing all users with less than 300 ratings, and all animes with less than 2500 ratings. 
<br>This leaves us with 4326 users, and 694 anime. This leaves us with 1M reviews.
<br> **KNN Collaborative Filter**
<br>Anime: Fruits Basket
<br>Iteration 1: Fill in NaN's with 0:
<p>Recommendations for 120 ['Fruits Basket']:
<br>1: ['Ouran Koukou Host Club'], with distance of 0.373222052075192:
<br>2: ['Chobits'], with distance of 0.4624859112856201:
<br>3: ['Fullmetal Alchemist'], with distance of 0.4802540588214821:
<br>4: ['D.N.Angel'], with distance of 0.484765127405153:
<br>5: ['Vampire Knight'], with distance of 0.4859228901227741:
<br>6: ['Sen To Chihiro No Kamikakushi'], with distance of 0.4903165145454421:
<br>7: ['Lovely★Complex'], with distance of 0.4960176951616878:
<br>8: ['Tsubasa Chronicle'], with distance of 0.49802071811742765:
<br>9: ['Suzumiya Haruhi No Yuuutsu'], with distance of 0.49827088077896575:
<br>10: ['Full Metal Panic!'], with distance of 0.4997038875693144:</p>

<br>Iteration 2: Fill in NaN's with average user rating:
<p>Recommendations for 120 ['Fruits Basket']:
<br>1: ['Skip Beat!'], with distance of 0.373222052075192:
<br>2: ['Howl No Ugoku Shiro'], with distance of 0.4624859112856201:
<br>3: ['Absolute Duo'], with distance of 0.4802540588214821:
<br>4: ['Zoku Natsume Yuujinchou'], with distance of 0.484765127405153:
<br>5: ['Lovely★Complex'], with distance of 0.4859228901227741:
<br>6: ['Natsume Yuujinchou'], with distance of 0.4903165145454421:
<br>7: ['Nodame Cantabile'], with distance of 0.4960176951616878:
<br>8: ['Natsume Yuujinchou San'], with distance of 0.49802071811742765:
<br>9: ['Natsume Yuujinchou Shi'], with distance of 0.49827088077896575:
<br>10: ['Kimi Ni Todoke'], with distance of 0.4997038875693144:</p>

<br>Iteration 3: Fill in NaN's with average anime rating:
<p>Recommendations for 120 ['Fruits Basket']:
<br>1: ['Ouran Koukou Host Club'], with distance of 0.373222052075192:
<br>2: ['Vampire Knight'], with distance of 0.4624859112856201:
<br>3: ['07-Ghost'], with distance of 0.4802540588214821:
<br>4: ['Lovely★Complex'], with distance of 0.484765127405153:
<br>5: ['Special A'], with distance of 0.4859228901227741:
<br>6: ['Vampire Knight Guilty'], with distance of 0.4903165145454421:
<br>7: ['Kamisama Hajimemashita'], with distance of 0.4960176951616878:
<br>8: ['Cardcaptor Sakura'], with distance of 0.49802071811742765:
<br>9: ['Howl No Ugoku Shiro'], with distance of 0.49827088077896575:
<br>10: ['D.N.Angel'], with distance of 0.4997038875693144:</p>


**Flash App**
-Use the anime_full['image_url'] column which has links to all anime photos, checked out url and it did not, but try later

## Next Steps & Conclusion
<br>-Find more user metadata to explore clusters of users.
<br>-Scrape description of each anime and user as a feature through NLP.
<br>-N-grams for the genre, pairs may be useful.

### Data Sources:
Anime & user metadata from : https://www.kaggle.com/azathoth42/myanimelist
Anime and rating data from: https://www.kaggle.com/CooperUnion/anime-recommendations-database



