# Anime_Recommender

The Data:
-anime_df: 12,294 animes
-rating_df: 7M reviews of 11,200 animes from 73,515 users
-anime_meta: 14,478 animes
-users_meta: 302,673 unique users
-Optional: concepts_titlesonly: 391706 anime titles & concepts, but no anime_id, so will be difficult to group with current dataset

**Content Based Recommender System:**
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


