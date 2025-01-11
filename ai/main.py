import re
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
import torch
from sklearn.metrics.pairwise import cosine_similarity

# 키워드 및 규칙 정의
keywords = ["매콤한", "짭짤한", "담백한", "고소한", "달콤한", "얼큰한", "칼칼한", "시원한", "새콤한", "쌉쌀한"]

keyword_rules = {
    "매콤한": ["매운", "스파이시", "불맛", "고추장", "양념"],
    "짭짤한": ["간장", "소금", "조림", "양념장"],
    "담백한": ["순한", "깔끔한", "클래식", "심플", "전통"],
    "고소한": ["참기름", "깨소금", "버터", "치즈", "견과"],
    "달콤한": ["꿀", "설탕", "시럽", "달콤", "스위트"],
    "얼큰한": ["국물", "탕", "찌개", "매운탕"],
    "칼칼한": ["얼큰한", "화끈한", "매운맛", "강렬한"],
    "시원한": ["동치미", "냉", "아이스", "맑은"],
    "새콤한": ["레몬", "유자", "식초", "과일", "산뜻"],
    "쌉쌀한": ["녹차", "말차", "다크", "커피", "씁쓸한"]
}

# 동의어 사전
synonyms = {
    "매운": ["맵다", "얼얼하다", "강렬한"],
    "짭짤한": ["짠맛", "간간한", "소금기"],
    "달콤한": ["달달한", "단맛", "달다"],
    "고소한": ["견과류", "치즈맛", "담백한 맛"]
}

# 텍스트 전처리
def preprocess_food_name(food_name):
    food_name = food_name.lower()  # 소문자 변환
    food_name = re.sub(r"[^가-힣a-z0-9\s]", "", food_name)  # 특수문자 제거
    return food_name


# 동의어를 이용한 규칙 확장
def expand_rules_with_synonyms(keyword_rules, synonyms):
    expanded_rules = {}
    for keyword, triggers in keyword_rules.items():
        expanded_triggers = set(triggers)
        for trigger in triggers:
            if trigger in synonyms:
                expanded_triggers.update(synonyms[trigger])
        expanded_rules[keyword] = list(expanded_triggers)
    return expanded_rules

expanded_rules = expand_rules_with_synonyms(keyword_rules, synonyms)


# 규칙 기반 키워드 매핑
def map_keywords_by_rules(food_name, keyword_rules):
    matched_keywords = []
    for keyword, triggers in keyword_rules.items():
        for trigger in triggers:
            if trigger in food_name:
                matched_keywords.append(keyword)
                break
    return matched_keywords

# BERT 모델 로드
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
bert_model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(keywords)
)

# BERT 기반 예측
def predict_keywords_with_bert(food_name, tokenizer, model):
    inputs = tokenizer(food_name, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = torch.sigmoid(outputs.logits).detach().numpy()
    return probabilities

# 규칙 기반과 BERT 결합
def hybrid_keyword_prediction(food_name, keyword_rules, synonyms, tokenizer, model):
    # 1. 전처리
    food_name = preprocess_food_name(food_name)

    # 2. 규칙 기반 매핑
    expanded_rules = expand_rules_with_synonyms(keyword_rules, synonyms)
    rule_based_keywords = map_keywords_by_rules(food_name, expanded_rules)

    # 3. BERT로 문맥 기반 확률 계산
    bert_probabilities = predict_keywords_with_bert(food_name, tokenizer, model)

    # 4. 규칙 기반과 BERT 결과 결합
    final_keywords = set(rule_based_keywords)
    for i, prob in enumerate(bert_probabilities[0]):
        if prob > 0.5:  # 임계값 설정
            final_keywords.add(keywords[i])

    return list(final_keywords)

# 테스트 음식명
test_foods = [
    "매운 갈비찜",
    "치즈 돈까스",
    "달콤한 초콜릿 케이크",
    "순두부 찌개",
    "고추장 불고기"
]

# 결과 출력
for food in test_foods:
    result = hybrid_keyword_prediction(food, keyword_rules, synonyms, tokenizer, bert_model)
    print(f"음식명: {food} → 키워드: {result}")