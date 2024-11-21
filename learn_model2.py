import pandas as pd
import re
import joblib
import os
import pyperclip  # Для работы с буфером обмена
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def load_csv_file(prompt):
    path = input(prompt)
    data = pd.read_csv(path, sep=";", header=None)
    return data[0].tolist(), path

def augment_data_with_regex(data, trigger_words):
    augmented_data = []
    for phrase in data:
        regex_hits = sum(1 for word in trigger_words if re.search(rf'\b{word}\b', phrase, re.IGNORECASE))
        augmented_data.append((phrase, regex_hits))
    return augmented_data

def main():
    # Загрузка данных
    malicious_data, _ = load_csv_file("Введите путь к CSV файлу с мошенническими фразами: ")
    normal_data, _ = load_csv_file("Введите путь к CSV файлу с обычными фразами: ")
    trigger_words, trigger_path = load_csv_file("Введите путь к CSV файлу с триггерными словами: ")
    
    # Создание обучающей выборки
    malicious_augmented = augment_data_with_regex(malicious_data, trigger_words)
    normal_augmented = augment_data_with_regex(normal_data, trigger_words)
    
    all_phrases = [item[0] for item in malicious_augmented + normal_augmented]
    regex_features = [item[1] for item in malicious_augmented + normal_augmented]
    labels = [1] * len(malicious_augmented) + [0] * len(normal_augmented)
    
    # Векторизация текста
    vectorizer = TfidfVectorizer()
    text_features = vectorizer.fit_transform(all_phrases)
    
    # Добавление регулярных признаков
    import numpy as np
    features = np.hstack([text_features.toarray(), np.array(regex_features).reshape(-1, 1)])
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Обучение модели
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Оценка модели
    y_pred = model.predict(X_test)
    print("Качество модели:")
    print(classification_report(y_test, y_pred))
    
    # Сохранение модели в папку триггеров
    model_path = os.path.join(os.path.dirname(trigger_path), "custom_regex_model.pkl")
    joblib.dump((model, vectorizer), model_path)
    print(f"Модель сохранена по пути: {model_path}")
    
    # Копирование пути в буфер обмена
    pyperclip.copy(model_path)
    print("Путь к модели скопирован в буфер обмена.")

if __name__ == "__main__":
    main()
