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

from recommender_system_base import RecommenderSystemBase
class HybridRecommenderSystem(RecommenderSystemBase):
    """
    Attributes
    ----------
    Methods
    -------
    compute_movie_embeddings
        Computes the movie embeddings.
    recommend_similar_movies
        Recommends the k most similar of the movie with the id 'movie_id'.
    recommend_movies_to_user
        Given a user with a watch history, it recommends the k movies that he will most likely watch.
    get_movies_embeddings
        Returns the embedding of the movies with movie_id in movie_ids.
   """

    def __init__(self, ratings_dataframe: pandas.DataFrame, movies_metadata_dataframe: pandas.DataFrame,
                 keywords_dataframe: pandas.DataFrame) -> None:
    
        super().__init__(ratings_dataframe, movies_metadata_dataframe, keywords_dataframe)
        # FIXME
        
    def recommend_similar_movies(self, movie_id: str, k: int) -> pandas.DataFrame:
        """Recommends the k most similar movies of the movie with the id 'movie_id'.
        Parameters
        ----------
        movie_id : str
            The id of the movie.
        k : int
            The number of similar movies to recommend.
        Returns
        -------
        pandas.DataFrame
            A subset of the movies_dataframe with the k similar movies of the target movie (movie_id).
        """
        content_df= self.movies_dataframe
        def movie_to_row(given_movie_id, movie_index_id):
            temp=0
            for key, value in movie_index_id.items():
                if value == given_movie_id:
                    return key
            temp+=1
            if temp!=0:return False
        
        def row_movie(given_row,movie_index_id ):
            temp=0
            for key, value in movie_index_id.items():
                if key == given_row:
                    return value
            temp+=1
            if temp!=0:return False
        given_movie_id = content_df[content_df['movie_id']== movie_id]['id'].values
        movie_index_id=self.movie_row_id
        row_val= movie_to_row(given_movie_id, movie_index_id)
        if row_val is not bool:
            x= self.cosine_similarity_tfidf
            element= x.iloc[row_val].tolist()
            index_numbers = np.argsort(element).tolist()
            l= len(index_numbers)
            row_numbers= index_numbers[l-k-1:l-1]
            out_name=list()
            for i in range(0,len(row_numbers)):
                ind= row_movie(row_numbers[i],movie_index_id )
                out_name.append(content_df[content_df['id']== ind]['movie_id'].values)
                
            df = pd.DataFrame(columns = out_name)
            return(df)
        else: 
            print('Gien movie {} was not in dataset or keywords were not there'.format(movie_id))
            y=pd.DataFrame(columns = [given_movie_id])
            return (y)  
        
    def get_movies_embeddings(self, movie_ids: [str]) -> pandas.DataFrame:
        """Returns the embedding of the movies with movie_id in movie_ids.
        Parameters
        ----------
        movie_ids : [str]
            List of the movies movie_id.
        Returns
        -------
        pandas.DataFrame
            The embeddings of the movies with movie_id in movie_ids.
        """
        content_df= self.movies_dataframe
        movie_index_id=self.movie_row_id
        encoding= self.encoded_tfidf
        
        def movie_to_row(given_movie_id, movie_index_id):
            temp=0
            for key, value in movie_index_id.items():
                if value == given_movie_id:
                    return key
            temp+=1
            if temp!=0:return False

        row_numbers=list()
        not_found=list()
        for i in movie_ids:
            given_movie_id = content_df[content_df['movie_id']== i]['id'].values
            row_val= movie_to_row(given_movie_id, movie_index_id)
            if row_val is not bool:
                row_numbers.append(movie_to_row(given_movie_id, movie_index_id))
            else:not_found.append(i)
        
        if len(not_found)!=0:
            for i in not_found:
                print('movie name not matching {}'.format(i))
                movie_ids.remove(i) 
                
        dfencoded= encoding.iloc[row_numbers,:]
        dfencoded.insert(0, 'movie_id', movie_ids)
        
        return (dfencoded)
    
    def recommend_movies_to_user(self, user_id: int, k: int) -> pandas.DataFrame:
        x_usr = []
        x_mov = []
        ratings = self.ratings_dataframe
        movies = self.movies_dataframe
        user_id_enc = int(self.user_encoded.get(user_id))
        unwatched = movies[~movies["id"].isin(ratings[ratings.userId==user_id]["movieId"])]["id"].values
        for i in unwatched:
            if('-' not in i):  
                i = int(i)
            if(self.movie_encoded.get(i) != None): 
                    x_usr.append(user_id_enc)
                    x_mov.append(self.movie_encoded.get(i))

        x_usr = np.array(x_usr)
        x_mov = np.array(x_mov)  
        user_pred_ratings = self._collaborative_model.predict([x_usr,x_mov])
        x_mov = np.expand_dims(x_mov,1)
        user_pred_ratings = np.hstack((x_mov,user_pred_ratings))
        sorted_ratings = user_pred_ratings[user_pred_ratings[:,1].argsort()[::-1]]
        watched = ratings[ratings.userId==user_id]
        
        recommended_by_system = []
        
        for i in sorted_ratings[:k]:
            #if(not watched[watched["movieId"] == self.movie_decoded.get(i[0])]["movieId"].values):
            a = movies[movies["id"] == str(self.movie_decoded.get(i[0]))]
            recommended_by_system.append(a["movie_id"].values[0])

        return movies[movies["movie_id"].isin(recommended_by_system)]

