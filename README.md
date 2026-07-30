# NYC Rental Price Prediction

## Описание проекта

Машинное обучение для предсказания стоимости аренды квартир в Нью-Йорке на основе характеристик жилья.

**Цель:** Построить модель регрессии, которая предсказывает цену аренды по 22 признакам (количество комнат, удобства, локация).

## Задачи проекта

- Реализовать линейную регрессию с нуля (аналитическое решение + градиентный спуск)
- Сравнить с sklearn реализацией
- Добавить регуляризацию (Ridge, Lasso, ElasticNet)
- Нормализация признаков (MinMax, Standard)
- Feature engineering (полиномиальные признаки)
- Анализ стабильности моделей

## Технологии

- **Python 3.12**
- **Библиотеки:**
  - `numpy` - матричные операции
  - `pandas` - работа с данными
  - `scikit-learn` - ML алгоритмы
  - `matplotlib` - визуализация

## Структура проекта

ml_project/<br>
│<br>
├── README.md<br>
├── requirements.txt<br>
├── .gitignore<br>
├── main.py<br>
│<br>
├── dataset/<br>
│   ├── raw/<br>
│   │   └── train.json<br>
│   └── processed/<br>
│       ├── train_clean.json<br>
│       └── test_clean.json<br>
│<br>
├── notebooks/<br>
│   ├── 01_EDA.ipynb                    # Разведочный анализ<br>
│   ├── 02_feature_engineering.ipynb     # Признаки + feature selection (SHAP, permutation)<br>
│   ├── 03_modeling.ipynb               # Linear → RF → XGBoost → сравнение<br>
│   └── 04_hyperparameter_tuning.ipynb   # GridSearch, RandomSearch, Optuna<br>
│<br>
└── src/<br>
    ├── load.py          # Загрузка + очистка выбросов + time-based split<br>
    ├── preprocess.py    # Feature engineering (amenities, time, encoding)<br>
    ├── selection.py     # Feature selection (Lasso L1, SHAP, permutation importance)<br>
    ├── models.py        # Custom + sklearn + RF + XGBoost + скалеры<br>
    ├── evaluate.py      # Метрики + 5 стратегий CV (KFold, Group, Stratified, TimeSeries)<br>
    └── train.py         # Обучение + GridSearch + Optuna + сохранение<br>


## Инструкция по запуску

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/Si1ice/nyc-rental-prediction.git
cd nyc-rental-prediction
```
### 2.Создайте виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows
```
### 3.Установка библиотек
```bash
pip install -r requirements.txt
```

### Откройте notebooks/01_EDA.ipynb и следуйте инструкциям.

## Дальнейшие планы
- Добавить более сложные модели (Random Forest, XGBoost)
- Feature selection (SHAP, permutation importance)
- Cross-validation
- Hyperparameter tuning (GridSearch, Optuna)
- Docker контейнеризация
- MLflow для экспериментов
- Deployment (Flask/FastAPI)

# Автор
[Булат Тарисов]<br>
GitHub: @Si1ice<br>
LinkedIn: [В разработке]<br>
Email: tarisovb@yandex.ru<br>