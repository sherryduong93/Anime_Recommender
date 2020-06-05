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

anime_df = pd.read_csv('data/anime.csv')
rating_df = pd.read_csv('data/rating.csv')
anime_meta = pd.read_csv('data/AnimeList_meta.csv')
users_meta = pd.read_csv('data/UserList_Meta.csv')
rating_df = rating_df[rating_df['rating']!=-1]

anime_full = full_anime_df(rating_df, anime_df, anime_meta)
anime_map = anime_full[['anime_id','name','title_english', 'type']]

train, test = all_spark.randomSplit([0.8, 0.2], seed=0)
train_data, val_data = train.randomSplit([0.8, 0.2], seed=0)

als_model = ALS(
    itemCol='anime_id',
    userCol='user_id',
    ratingCol='rating',
    nonnegative=True,    
    maxIter=20,
    regParam=0.1,
    rank=10) 
als_model.setColdStartStrategy("drop")

predictions = recommender.transform(test_data)
predictions_df = predictions.toPandas()
rmse = np.sqrt(mean_squared_error(predictions_df['rating'],predictions_df['prediction']))
pct_mean = round(rmse/(rating_df['rating'].mean())*100,0)
print(f'rmse is {pct_mean}% of mean')
os.system("say 'Complete'") 

