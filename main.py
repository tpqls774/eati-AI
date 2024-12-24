import re
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

keywords = ['담백한', '칼칼한', '매콤한', '고소한', '짭짤한', '얼큰한', '시원한', '새콤한', '달달한', '쌉쌀한']

keyword_rules = {
    "담백한": ["순한", "깔끔한", "클래식", "심플", "전통"],
    "칼칼한": ["얼큰한", "화끈한", "매운맛", "강렬한"],
    "매콤한": ["매운", "양념", "고추", "불맛", "핫", "김치"],
    "고소한": ["치즈", "참깨", "참기름", "버터"],
    "짭짤한": ["간장", "소금", "조림"],
    "얼큰한": ["국물", "매운탕", "탕", "찌개"],
    "시원한": ["동치미", "냉면", "냉국", "아이스", "맑은"],
    "새콤한": ["레몬", "유자", "신", "과일", "산뜻"],
    "달달한": ["꿀", "설탕", "시럽", "달콤", "스위트", "케이크"],
    "쌉쌀한": ["녹차", "말차", "다크", "커피"]
}

def map_keywords_by_rules(food_name, keyword_rules):
    # 결과 키워드 저장 리스트
    matched_keywords = []
    # 각 키워드 규칙 확인
    for keyword, triggers in keyword_rules.items():
        for trigger in triggers:
            if trigger in food_name:  # 음식명에 트리거 단어가 포함되면
                matched_keywords.append(keyword)
                break
    return matched_keywords

# 테스트 데이터
food_names = ["매운 갈비찜", "치즈 돈까스", "딸기 케이크", "순두부 찌개"]

data = []
# 결과 출력
for food in food_names:
    vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    keyword = map_keywords_by_rules(food, keyword_rules)
    keyword = keyword[0]
    for i in range(len(keywords)):
        if keyword in keywords[i]:
            vector[i] = 1
    data.append((food, vector))

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

input_ids = []
attention_masks = []
labels = []

for text, label in data:
    tokenized = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        padding='max_length',
        truncation=True
    )
    input_ids.append(tokenized['input_ids'])
    attention_masks.append(tokenized['attention_mask'])
    labels.append(label)

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels, dtype=torch.float32)

train_dataset = TensorDataset(input_ids, attention_masks, labels)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(keywords))

optimizer = optim.AdamW(model.parameters(), lr=1e-5)

epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, label = batch
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# 새로운 음식명 예측
test_input = tokenizer("제주 흑돼지 돈까스", return_tensors="pt", padding=True, truncation=True, max_length=128)
model.eval()
with torch.no_grad():
    outputs = model(**test_input)
    probabilities = torch.sigmoid(outputs.logits)  # 확률값 출력

# Threshold를 사용해 최종 키워드 선택
threshold = 0.5
predicted_labels = (probabilities > threshold).int()
predicted_keywords = [keywords[i] for i, value in enumerate(predicted_labels[0]) if value == 1]

print("예측된 키워드:", predicted_keywords)