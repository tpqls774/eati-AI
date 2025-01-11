from transformers import AutoTokenizer, BertForSequenceClassification, get_scheduler
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

df = pd.read_csv("labels_data.csv")

# 데이터셋 클래스 정의
class FoodDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row['음식명']
        labels = row.iloc[1:].values.astype(float)  # 레이블 (키워드 0/1)
        # 토큰화
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

# Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
dataset = FoodDataset(df, tokenizer, max_len=64)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 키워드 및 규칙 정의
keywords = ["매콤한", "짭짤한", "담백한", "고소한", "달콤한", "얼큰한", "칼칼한", "시원한", "새콤한", "쌉쌀한"]

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


# BERT 모델 로드
num_labels = len(df.columns) - 1  # 키워드 개수
model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=num_labels
)

# 옵티마이저와 손실 함수 정의
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.BCEWithLogitsLoss()  # 다중 레이블 분류용 손실 함수

# 학습 스케줄러
num_training_steps = len(dataloader) * 5  # 에포크 5 기준
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# GPU 사용 여부
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# 학습 루프
epochs = 5
for epoch in range(epochs):
    model.train()
    loop = tqdm(dataloader, leave=True)
    for batch in loop:
        # 데이터 준비
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 모델 예측 및 손실 계산
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 진행 상황 표시
        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

def evaluate_model(model, dataloader, threshold=0.5):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits)  # 확률 계산
            preds = (probs > threshold).float()  # 임계값으로 이진화

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # F1 스코어 및 정확도 계산
    f1 = f1_score(true_labels, predictions, average="macro")
    accuracy = accuracy_score(true_labels, predictions)
    return f1, accuracy

# 평가 실행
f1, accuracy = evaluate_model(model, dataloader)
print(f"F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")

model.save_pretrained("fine_tuned_bert")
tokenizer.save_pretrained("fine_tuned_bert")