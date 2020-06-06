import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from src.data_funcs import *
from src.model_funcs import *
from pyspark.sql.functions import explode
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.getOrCreate()

anime_df = pd.read_csv('data/anime.csv')
rating_df = pd.read_csv('data/rating.csv')
anime_meta = pd.read_csv('data/AnimeList_meta.csv')
users_meta = pd.read_csv('data/UserList_Meta.csv')
rating_df = rating_df[rating_df['rating']!=-1]

anime_full = full_anime_df(rating_df, anime_df, anime_meta)
anime_map = anime_full[['anime_id','name','title_english', 'type']]

all_spark = spark.createDataFrame(rating_df)

train, test = all_spark.randomSplit([0.8, 0.2], seed=0)

class Anime_RecommenderCF():

    def fit(self, train_spark):
        self.als_model = ALS(
            itemCol='anime_id',
            userCol='user_id',
            ratingCol='rating',
            nonnegative=True,    
            maxIter=20,
            regParam=0.1,
            rank=10) 
        self.als_model.setColdStartStrategy("drop")
        self.recommender = (self.als_model.fit(train_spark))
        self.movieRecs = self.recommender.recommendForAllItems(10)
        self.userRecs = self.recommender.recommendForAllUsers(10)

    def transform(self, test_spark):
        self.test_spark = test_spark
        return (self.als_model.transform(test))
    
    def predictions(self):
        return self.transform(self.test_spark).toPandas
    
    def rmse(self):
        df = self.predictions(self.test_spark)
        rmse_test = np.sqrt(mean_squared_error(df['rating'],df['prediction']))
        return rmse_test
    
    def other_user_recs(self, anime_id, anime_map):
        rec = self.movieRecs.filter(self.movieRecs.anime_id==anime_id)
        recs = rec.withColumn('recommendations', explode(rec.recommendations))
        sim_users = recs.select("anime_id", 'recommendations.*').select("user_id").rdd.flatMap(lambda x: x).collect()
        user_rec = self.userRecs.where(self.userRecs.user_id.isin(sim_users))
        user_recs = user_rec.withColumn('recommendations', explode(user_rec.recommendations))
        user_picks = user_recs.select("user_id", 'recommendations.*')
        userpicks_df = user_picks.toPandas()
        avg_rating = userpicks_df.groupby('anime_id').mean()['rating']
        count_rating = userpicks_df.groupby('anime_id').count()['rating']
        user_recs_joined = pd.DataFrame([avg_rating,count_rating],columns=avg_rating.index, index=['avg_rating','count_rating']).T
        user_recs_joined['weighted_avg'] = weighted_rating(user_recs_joined,'count_rating', 'avg_rating')
        top_anime_recs = user_recs_joined.sort_values('weighted_avg')[:10].index
        return anime_map[anime_map['anime_id'].isin(top_anime_recs)]

    def user_rec(self, user_id, anime_map):
        your_rec = self.userRecs.where(self.userRecs.user_id==user_id)
        yourrecs = your_rec.withColumn('recommendations', explode(your_rec.recommendations))
        your_picks = yourrecs.select("user_id", 'recommendations.*').select("anime_id").rdd.flatMap(lambda x: x).collect()
        return anime_map[anime_map['anime_id'].isin(your_picks)]




# rec = Anime_RecommenderCF()
# rec.fit(train)

# anime_id = int(input('enter anime ID of interest here'))
# user_id = int(input('enter user ID of interest here'))

# rec.other_user_recs(anime_id, anime_map)
# rec.user_rec(user_id,anime_map)