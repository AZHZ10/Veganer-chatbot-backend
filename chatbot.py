import pandas as pd
from sentence_transformers import SentenceTransformer
# 챗봇의 대화 유사도 개선을 위해 cosine_similarity 사용
from sklearn.metrics.pairwise import cosine_similarity
import json


def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model


def get_dataset():
    file_name = "data_file.csv"
    df = pd.read_csv(file_name)
    df['embedding'] = df['embedding'].apply(json.loads)
    return df


model = cached_model()
df = get_dataset()


def embedding_user_input(user_input):
    embedding = model.encode(user_input)
    return embedding


def get_answer(user_input):
    similarity_arr = []
    embedding = embedding_user_input(user_input)
    similarity_arr = df['embedding'].map(  # similarity 열을 embedding과의 cosine_similarity로 추가
        lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[similarity_arr.idxmax()]  # similarity가 가장 높은 행
    return answer['챗봇']
