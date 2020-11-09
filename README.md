# Hybrid-Recommended-Systems


 The Class RecommenderSystemBase has the functions that create instance for the init:
 
 model for the neural network: collabrative Recommmender model (_collaborative_model), content Recommender model( _content_model)
 
 Embeddings: embeddings usinf tfidf(__create_dataframe_tfidf), Collabrative Filtering (_colab_rat_encoding.) 
 
 and creationof Data Frame (def __create_movies_dataframe) and cosine similarity (__cosine_similarity_matrix)
 
 
 This class is taken to input in our classe Hyper Recomender system Super__init__ hence all instances get created at time of creation of object.
 
 The Hyper_Recomender
 has 3 functions:
 
 recommend_similar_movies : input is string that is tiltle(movie_id) in format (eg. ) and no. of recommended movies/content based approach 
 
 get_movies_embeddings  takes input a list contaning strings at each index that is title(movie_id) again in above format 
 
 recommend_movies_to_user  takes input as user if and no. of movies to recommend/ collabratie approach
 
 
 To run file please import all dependecies mentioned in file 
 '''
 
import pandas
import pandas as pd
import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import keras
from keras import layers
from keras.layers import Embedding, Input, dot, concatenate, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
'''


TO RUN
Datast avilabe on https://www.kaggle.com/rounakbanik/the-movies-dataset
download rattings, movies, keywords, credits in individual dataframes

# HOW TO RUN HELP
Object= HybridRecommenderSystem(rattings, movies, keywords, credits) 
#in this order note all are data frames
Object.recommend_similar_movies(movie_id: str, k:int)
movie_id title as   and k no. of time 
Object.get_movies_embeddings(movies:[str])
input a list in which strings are there
Object.recommend_movies_to_user(user_id: int, k:int)
#userid for which to be recommended and No.


Please note model might take time to run so lower the no. of Epochs in base file content_model function for a quick summary. 

# Refrences
https://www.kaggle.com/robottums/hybrid-recommender-systems-with-surprise
https://keras.io/examples/structured_data/collaborative_filtering_movielens/
https://keras.io/examples/structured_data/collaborative_filtering_movielens/

https://towardsdatascience.com/creating-a-hybrid-content-collaborative-movie-recommender-using-deep-learning-cc8b431618af

https://blog.keras.io/building-autoencoders-in-keras.html#:~:text=To%20build%20an%20autoencoder%2C%20you,a%20%22loss%22%20function



 
 
 
 
 
