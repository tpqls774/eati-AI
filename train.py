from transformers import AutoTokenizer, DistilBertForSequenceClassification, get_scheduler
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
        text = str(row['음식명'])
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

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # 데이터 로드
    df = pd.read_csv("labels_data.csv")
    train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

    # Tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    train_dataset = FoodDataset(train_data, tokenizer, max_len=64)
    val_dataset = FoodDataset(val_data, tokenizer, max_len=64)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=3, pin_memory=True)

    # BERT 모델 로드
    num_labels = len(df.columns) - 1  # 키워드 개수
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-multilingual-cased",
        num_labels=num_labels
    )

    # 옵티마이저와 손실 함수 정의
    optimizer = AdamW(model.parameters(), lr=3e-5)
    num_training_steps = len(train_loader) * 5  # 에포크 5 기준
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # GPU 사용 여부
    scaler = torch.amp.GradScaler()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    def evaluate_model(model, dataloader, threshold=0.7):
        model.eval()
        predictions = []
        true_lables = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float()

                predictions.extend(preds.cpu().numpy())
                true_lables.extend(labels.cpu().numpy())

        f1 = f1_score(true_lables, predictions, average="macro")
        accuracy = accuracy_score(true_lables, predictions)
        return f1, accuracy

    # 학습 루프
    epochs = 5
    best_val_f1 = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, leave=True)

        for step, batch in enumerate(loop):
            # 데이터 준비
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast('cuda'):
                # 모델 예측 및 손실 계산
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()

            # 역전파 및 최적화
            if (step + 1) % 4 == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # 진행 상황 표시
            train_loss += loss.item()
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

        val_f1, val_accuracy = evaluate_model(model, val_loader)
        print(f"Epoch {epoch} - Train Loss: {train_loss / len(train_loader):.4f} - Val F1: {val_f1:.4f} - Val Accuracy: {val_accuracy:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(f"best_model_epoch_{epoch}")
            tokenizer.save_pretrained(f"best_model_epoch_{epoch}")

    model.cpu()
    model.save_pretrained("final_model")
    tokenizer.save_pretrained("final_model")