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

class RecommenderSystemBase():

    def __init__(self, ratings_dataframe: pandas.DataFrame, movies_metadata_dataframe: pandas.DataFrame,
                 keywords_dataframe: pandas.DataFrame) -> None:
        
        
        self.movies_dataframe = self.__create_movies_dataframe(ratings_dataframe, movies_metadata_dataframe,
                                                                keywords_dataframe)
        self.ratings_dataframe = ratings_dataframe
        self.tfidf, self.movie_row_id = self.__create_dataframe_tfidf(self.movies_dataframe)
        self.encoded_tfidf = self._content_model(self.tfidf)
        self.cosine_similarity_tfidf= self.__cosine_similarity_matrix(self.encoded_tfidf)
        self.ratings_dataframe, self.movie_encoded, self.movie_decoded,self.user_encoded = self._colab_rat_encoding()
        self._collaborative_model = self._collaborative_model()
        
        

    def __create_movies_dataframe(self, ratings_dataframe, 
                                  movies_metadata_dataframe,
                                  keywords_dataframe):
        movies = movies_metadata_dataframe
        keyword= keywords_dataframe
        def json_format(line):
            line = ast.literal_eval(line)
            words = []
            for i in line:
                words.append(i["name"])
            words = " ".join(words)
            words = words
            return words
 
        def id_create(line):
            tagged =[]
            for i in range(len(line)):
                line[i]= line[i].replace(" ",'_')
                tagged.append(line[i])
            return "-".join(tagged)[:-6]
    
        movies = movies[~movies["title"].isna()]
        movies = movies.drop_duplicates(subset= ['id'])



        keywords["id"] = keywords["id"].astype(str) 
        content_df = movies.merge(keywords, on="id", how="inner")
        content_df = content_df.astype(str)
        content_df["genres"] = content_df["genres"].apply(lambda x :json_format(x) )
        content_df["keywords"] = content_df["keywords"].apply(lambda x :json_format(x) )
        content_df['keywords'] = content_df[['genres', 'keywords']].apply(lambda x: ' '.join(x), axis=1)
        content_df['movie_id'] =content_df[["title", "original_title", "original_language", "release_date"]].apply(lambda x: id_create(x), axis=1)
        content_df = content_df.drop_duplicates(subset= ['id'])
        
        return content_df
        
         
    def _colab_rat_encoding(self):
        ratings = self.ratings_dataframe
        def encoder_decoder(df, col):
            encoded = dict()
            count = 0
            for i in df[col].unique():
                encoded[i]=count
                count+=1

            decoded = dict()
            count = 0
            for i in df[col].unique():
                decoded[count]=i
                count+=1   
            return encoded, decoded
        
        movie_encoded, movie_decoded = encoder_decoder(ratings,"movieId")
        user_encoded, user_decoded = encoder_decoder(ratings,"userId")
        ratings["mov_encode"] = ratings["movieId"].map(movie_encoded)
        ratings["usr_encode"] = ratings["userId"].map(user_encoded)
        return ratings, movie_encoded, movie_decoded, user_encoded  
    
    
    def _collaborative_model(self):
        ratings = self.ratings_dataframe
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(ratings[["usr_encode","mov_encode"]].to_numpy(), ratings["rating"].to_numpy(), test_size=0.15, random_state=42)
        X_train.shape, X_test.shape, y_train.shape, y_test.shape

        users_n = len(ratings["userId"].unique())
        movies_n = len(ratings["movieId"].unique()) 
        # Emmbedding of Users
        usr_inp = Input(shape=(1,), name='usr_inp')
        usr_embed = Embedding(input_dim = users_n, output_dim=50, input_length=1)(usr_inp)
        usr_embed = Flatten(name='usr_embeddings_1')(usr_embed)

        # Embeddings of Movies
        mov_input = Input(shape=(1,), name='mov_input')
        mov_embed = Embedding(input_dim = movies_n, output_dim=50, input_length=1) (mov_input)
        mov_embed = Flatten(name='mov_embeddings_1') (mov_embed)

        merged_vectors = dot([usr_embed, mov_embed], name='Dot_Product', axes=1)

        model = Model([usr_inp, mov_input], merged_vectors)
        model.compile(loss='mean_squared_error', optimizer = Adam(learning_rate = 0.0005))
        model.fit( x=[X_train.T[0],X_train.T[1]],y=y_train, batch_size=32, epochs=38, verbose=1, callbacks= EarlyStopping(patience=5,monitor="val_loss"), validation_data=([X_test.T[0],X_test.T[1]], y_test))

        return model
     
    def __create_dataframe_tfidf(self, movies_dataframe):
        from sklearn.feature_extraction.text import TfidfVectorizer
        df = movies_dataframe
        tfidf = TfidfVectorizer(
                ngram_range=(0, 1),
                min_df=0.0001,
                stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['keywords'])
        movie_list=df['id'].tolist()
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df['id'].tolist())

        movie_index_id=dict()
        for i in range(0,len(movie_list)):
            movie_index_id[i]=movie_list[i]
        return(tfidf_df, movie_index_id)
    
    def _content_model(self, tfidf_df):
        #print(x_data.shape)
        # This is the size of our encoded representations
        x_data = tfidf_df
        shape0=x_data.shape[0]
        shape1=x_data.shape[1]
        encoding_dim = 100

         # This is our input image
        input_tfidf = keras.Input(shape=(shape1,))
        # "encoded" is the encoded representation of the input
        encoded = layers.Dense(1000, activation='relu')(input_tfidf)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)


        # "decoded" is the lossy reconstruction of the input
        decoded = layers.Dense( 1000, activation='relu')(encoded)
        decoded = layers.Dense(shape1 , activation='sigmoid')(decoded)

         # This model maps an input to its reconstruction
        autoencoder = keras.Model(input_tfidf, decoded)

        encoder = keras.Model(input_tfidf, encoded)


        autoencoder.compile(optimizer= Adam(learning_rate=0.001), loss='binary_crossentropy')
        autoencoder.summary()
        autoencoder.fit(x_data,x_data,
        epochs=20,
        batch_size=256,
        shuffle=False,
        validation_split=0.2
        )

        encoded_tfidf = encoder.predict(x_data)
        data_frame= pd.DataFrame(encoded_tfidf)
        return(data_frame)
    
    def __cosine_similarity_matrix(self, encoded_tfidf):
        encoded= encoded_tfidf
        from sklearn.metrics.pairwise import cosine_similarity
        x= pd.DataFrame(cosine_similarity(encoded))
        return (x)
        
        
