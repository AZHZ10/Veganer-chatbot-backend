import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json


class ChatModel():
    def __init__(self):
        self.initialize()

    def initialize(self, ):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('jhgan/ko-sroberta-multitask')

        self.file_name = "data_file.csv"
        self.df = pd.read_csv(self.file_name)
        self.df['embedding'] = self.df['embedding'].apply(json.loads)

    def embedding_user_input(self, user_input):
        embedding = self.model.encode(user_input)
        return embedding

    def get_answer(self, user_input):
        similarity_arr = []
        embedding = self.embedding_user_input(user_input)
        similarity_arr = self.df['embedding'].map(  # similarity 열을 embedding과의 cosine_similarity로 추가
            lambda x: cosine_similarity([embedding], [x]).squeeze())
        answer = self.df.loc[similarity_arr.idxmax()]  # similarity가 가장 높은 행
        return answer['Answer']
