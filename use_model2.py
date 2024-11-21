import joblib
import pandas as pd
import re
import numpy as np

def load_csv_file(prompt):
    path = input(prompt)
    data = pd.read_csv(path, sep=";", header=None)
    return data[0].tolist()

def load_model():
    path = input("Введите путь к модели (custom_regex_model.pkl): ")
    try:
        return joblib.load(path), path
    except FileNotFoundError:
        print("Ошибка: модель не найдена. Убедитесь, что файл существует.")
        exit()

def augment_phrase_with_regex(phrase, trigger_words):
    return sum(1 for word in trigger_words if re.search(rf'\b{word}\b', phrase, re.IGNORECASE))

def main():
    # Загрузка модели и триггеров
    (model, vectorizer), model_path = load_model()
    trigger_words = load_csv_file("Введите путь к CSV файлу с триггерами (точка с запятой): ")
    
    print(f"Модель загружена из: {model_path}")
    
    while True:
        phrase = input("Введите фразу для анализа (или 'exit' для выхода): ")
        if phrase.lower() == "exit":
            break
        
        # Обработка фразы
        text_feature = vectorizer.transform([phrase]).toarray()
        regex_feature = augment_phrase_with_regex(phrase, trigger_words)
        features = np.hstack([text_feature, np.array([regex_feature]).reshape(-1, 1)])
        
        # Прогноз
        prediction = model.predict(features)
        confidence = model.predict_proba(features).max()
        
        # Вывод результата
        if prediction[0] == 1:
            print(f"Мошенническая фраза с уверенностью {confidence:.2f}")
        else:
            print(f"Обычная фраза с уверенностью {confidence:.2f}")

if __name__ == "__main__":
    main()
