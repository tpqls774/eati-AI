import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from collections import Counter
import re

# 모델 및 토크나이저 로드
model_path = "final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()  # 평가 모드
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# 키워드 리스트 (학습에 사용한 키워드)
keywords = ["매콤한", "짭짤한", "담백한", "고소한", "달달한", "얼큰한", "칼칼한", "시원한", "새콤한", "쌉쌀한"]

# 데이터 예시 (가게 이름과 음식명)
data = [
    {"가게이름": "맛있는 김밥집", "음식명": "참치김밥"},
    {"가게이름": "맛있는 김밥집", "음식명": "치즈김밥"},
    {"가게이름": "맛있는 김밥집", "음식명": "라면"},
    {"가게이름": "맛있는 김밥집", "음식명": "돈가스김밥"},
    {"가게이름": "맛있는 김밥집", "음식명": "우동"},
    {"가게이름": "감성 카페", "음식명": "아메리카노"},
    {"가게이름": "감성 카페", "음식명": "카페라떼"},
    {"가게이름": "감성 카페", "음식명": "바닐라라떼"},
    {"가게이름": "감성 카페", "음식명": "아이스 아메리카노"},
    {"가게이름": "감성 카페", "음식명": "딸기라떼"},
]

# 데이터프레임으로 변환
df = pd.DataFrame(data)

# 키워드 예측 함수
def predict_keywords(food_name):
    # 음식명을 토큰화
    encoding = tokenizer(
        food_name,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # 모델 예측
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    # 임계값 0.5 이상인 키워드만 반환
    threshold = 0.5
    predicted_keywords = [keywords[i] for i, prob in enumerate(probs) if prob > threshold]
    return predicted_keywords

# 음식별 키워드 예측
df["음식키워드"] = df["음식명"].apply(predict_keywords)

# 가게별 키워드 집계
def aggregate_store_keywords(store_group):
    all_keywords = sum(store_group["음식키워드"], [])  # 모든 음식의 키워드 합치기
    keyword_counts = Counter(all_keywords)  # 키워드 빈도수 계산
    return keyword_counts.most_common()  # 키워드 순위 반환

store_keywords = df.groupby("가게이름").apply(aggregate_store_keywords)

# 결과 출력
print("음식별 키워드:")
print(df)

print("\n가게별 키워드:")
for store, keywords in store_keywords.items():
    print(f"{store}: {keywords}")

# 수정 예정