import pandas as pd
import torch
import re
from sentence_transformers import SentenceTransformer, util

# 데이터셋 로드
df = pd.read_csv("data.csv")

df = df.dropna(subset=['음식명'])
df["음식명"] = df["음식명"].astype(str)

# 키워드와 규칙 정의
keywords = ["매콤한", "짭짤한", "담백한", "고소한", "달달한", "얼큰한", "칼칼한", "시원한", "새콤한", "쌉쌀한"]

def preprocess(food_name):
    food_name = food_name.lower()
    food_name = re.sub(r"[^가-힣a-z0-9\s]", "", food_name)
    return food_name

df["음식명"] = df["음식명"].apply(preprocess)

# 가장 유사한 k개의 키워드로 레이블 생성
def top_keywords(similarity_scores, k=4):
    sorted_indices = torch.topk(torch.tensor(similarity_scores), k=k).indices
    labels = [1 if i in sorted_indices else 0 for i in range(len(similarity_scores))]
    return labels

# Sentence-BERT 모델 로드
model = SentenceTransformer('all-mpnet-base-v2')

# 음식명과 키워드 임베딩
food_embeddings = model.encode(df["음식명"].tolist(), convert_to_tensor=True)
keyword_embeddings = model.encode(keywords, convert_to_tensor=True)

# 유사도 계산
similarities = util.cos_sim(food_embeddings, keyword_embeddings)

# 가장 유사한 키워드로 레이블 생성
threshold = 0.7  # 유사도 임계값
df["레이블"] = similarities.cpu().numpy().tolist()  # 유사도를 레이블 형태로 변환

# 키워드별 레이블 컬럼 추가
for i, keyword in enumerate(keywords):
    df[keyword] = df["레이블"].apply(lambda x: top_keywords(x, k=4)[i])

# 불필요한 열 삭제
df = df.drop(columns=["레이블"])
print(df)
print(df["음식명"].isnull().sum())

# 데이터 생성
df.to_csv("labels_data.csv", index=False)