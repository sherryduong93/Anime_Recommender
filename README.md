# Anime_Recommender

The Data:
-anime_df: 12,294 animes
-rating_df: 7M reviews of 11,200 animes from 73,515 users
-anime_meta: 14,478 animes
-users_meta: 302,673 unique users
-Optional: concepts_titlesonly: 391706 anime titles & concepts, but no anime_id, so will be difficult to group with current dataset

Features from anime_df:
--Genre 
--How many episodes > feature engineer for "Commitment Type"
--Movie or TV as a feature > "Occasion Type"

Features from anime_meta:
-Source: Will users care if it is from a Manga or not?
-Status: Finished Airing: What does that mean?
-Duration (per episode)
-Rating (PG-13, etc)
-Licensor (11K missing values, ignore)
-Studio (6K missing values) > perhaps categorical for "hot studio" or not
-Producer (6.2K missing values) > perhaps categorical for "hot producer" or not

User metadata from users_meta:
-Gender
-Location
-Birthdate > convert to age
-Member since data > Age of membership
-Stats_episode > How active is the user?
-Stats mean score > How generous/strict is the user?
-username: Could we pull anything useful from the usernames? NLP?

## EDA
-avg_rating from the anime dataset: 6.473902
-weighted_rating average from the anime dataset: 6.654531
-Average ratings from the user rating dataset: 7.8
-Could be due to rating data missing some ratings of animes listed in the anime dataset


## Content Based Recommender System:
**Baseline Content Based Recommender:**
-Features: Type (Movie, TV, etc.), Source (Manga, Music, Book, etc.), Rating Type (PG, R, etc.), and Weighted Rated.
-Similarity Metrics: Tested Cosine Similarity & Pairwise Distance. Spot-checking a few popular animes in each genre, Pairwise Distance performs the best with genre related recommendations, but is recommending unpopular anime, cosine similarity is recommending the popular anime, but not good at narrowing down to the right genre. 
**Content Based Recommender Iteration 2:**
<br>-Added dummified genre to the 
<br>-Based on the EDA, some producers/studios have higher ratings overall than others, only the top 20 studios and producers will be captured in the next content based model.
<br>-Adding Studio/Producers had almost no impact on the similarity matrix of the iteration with only genre added.


**Data Cleaning To-Do's**
-Duration: Needs to be converted to minutes without str
-Genre: Vectorize and convert to features. Is it important to have more N-grams?
-Studio & Producer: Perhaps only certain producers have high scores, look into this. Then, only categorical based on the popular ones.
-Ratings: Before alteration the ratings dataset uses a "-1" to represent missing ratings. I'm replacing these placeholders with a null value because I will later be calculating the average rating per user and don't want the average to be distorted
-Opening & Ending Themes: NLP - Any interesting clusters?
-NOTE: Ratings matrix does not have ALL of the ratings provided by the users that make up the average rating column in the anime_df, will need to consider this for the collaborative filter based recommender. 

**Process**
-Clustering anime groups based on the above features
-Calculate similarity matrix on non-zero values

**Flash App**
-Use the anime_full['image_url'] column which has links to all anime photos, checked out url and it did not, but try later


