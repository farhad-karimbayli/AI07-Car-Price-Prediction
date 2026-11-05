"""
Задание 3: Подготовка данных и демонстрация методов кодирования/масштабирования
• Выделите признаки X и целевую переменную y.
• Создайте признак mileage_per_year — средний годовой пробег.
• Разделите данные: 80% для обучения, 20% для проверки, random_state=42.
• Заполните пропуски числовых признаков медианой, категориальных — наиболее частым значением.
• Примените One-Hot Encoding к категориальным признакам.
• Масштабируйте числовые признаки с помощью StandardScaler.
• Объедините обработку данных и модель через ColumnTransformer и Pipeline.
Важно: параметры определяются только по обучающим данным.

Отдельно на копии обучающих данных покажите:
• Результат factorize и One-Hot Encoding для столбца brand.
• Результат MinMaxScaler и StandardScaler для столбца mileage_km.
Объясните различия между этими способами преобразования.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


def demonstrate_transformations(X_train: pd.DataFrame):
    print("\n" + "-" * 60)
    print("ДЕМОНСТРАЦИЯ МЕТОДОВ ПРЕОБРАЗОВАНИЯ (на копии обучающих данных)")
    print("-" * 60)

    # 1. Демонстрация factorize vs One-Hot Encoding для brand
    demo_df = X_train.copy()

    # pd.factorize
    codes, uniques = pd.factorize(demo_df["brand"])
    factorize_series = pd.Series(codes, index=demo_df.index, name="brand_factorized")

    # One-Hot Encoding через pd.get_dummies
    ohe_df = pd.get_dummies(demo_df[["brand"]], prefix="brand", drop_first=False)

    compare_cat = pd.concat([demo_df[["brand"]], factorize_series, ohe_df], axis=1)
    print("\n1. Сравнение factorize и One-Hot Encoding для столбца 'brand' (первые 5 строк):")
    print(compare_cat.head())

    print("\n   [Различия между factorize и One-Hot Encoding]:")
    print("   • factorize: преобразует категории в порядковые целые числа (0, 1, 2...).")
    print("     Минус для линейных моделей: искусственно вводит ложный порядок и расстояния")
    print("     между брендами (например, BMW=0, Hyundai=1, Toyota=2 => Toyota > BMW).")
    print("   • One-Hot Encoding: создает отдельные бинарные индикаторы (0 или 1) для каждой")
    print("     категории. Линейная модель может назначить собственный независимый вес")
    print("     каждой марке автомобиля, не искажая природу признака.")

    # 2. Демонстрация MinMaxScaler vs StandardScaler для mileage_km
    mileage_demo = demo_df[["mileage_km"]].copy()
    # Временно заполним пропуски медианой для корректного масштабирования в демо
    mileage_demo = mileage_demo.fillna(mileage_demo.median())

    mms = MinMaxScaler()
    std = StandardScaler()

    mileage_mms = mms.fit_transform(mileage_demo)
    mileage_std = std.fit_transform(mileage_demo)

    compare_num = pd.DataFrame({
        "Исходный mileage_km": mileage_demo["mileage_km"].values,
        "MinMaxScaler [0, 1]": mileage_mms.flatten(),
        "StandardScaler (mean=0, std=1)": mileage_std.flatten(),
    }, index=demo_df.index)

    print("\n2. Сравнение MinMaxScaler и StandardScaler для 'mileage_km' (первые 5 строк):")
    print(compare_num.head().round(4))

    print("\n   [Различия между MinMaxScaler и StandardScaler]:")
    print("   • MinMaxScaler: сжимает данные строго в фиксированный диапазон [0, 1].")
    print("     Формула: (x - x_min) / (x_max - x_min). Чувствителен к выбросам, так как")
    print("     выбросы стягивают остальные значения в узкий интервал.")
    print("   • StandardScaler: центрирует данные относительно среднего и делит на std.")
    print("     Формула: (x - mean) / std. Итоговое распределение имеет mean = 0 и std = 1.")
    print("     Устойчив к выбросам и идеально подходит для линейной регрессии и оптимизаторов.")


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    # Пайплайн для числовых признаков:
    # 1. Заполнение пропусков медианой
    # 2. Масштабирование StandardScaler
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Пайплайн для категориальных признаков:
    # 1. Заполнение пропусков наиболее частым значением (most_frequent)
    # 2. One-Hot Encoding с handle_unknown='ignore'
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    return preprocessor


def run_task3(df: pd.DataFrame):
    print("=" * 65)
    print("ЗАДАНИЕ 3: Подготовка данных и построение ColumnTransformer")
    print("=" * 65)

    df_prepared = df.copy()

    # 1. Создание нового признака mileage_per_year
    df_prepared["mileage_per_year"] = df_prepared["mileage_km"] / df_prepared["age_years"]
    print("1. Создан признак 'mileage_per_year' (среднегодовой пробег):")
    print(df_prepared[["age_years", "mileage_km", "mileage_per_year"]].head())

    # 2. Выделение признаков X и целевой переменной y
    target_col = "price_azn"
    numeric_features = ["age_years", "mileage_km", "engine_l", "power_hp", "mileage_per_year"]
    categorical_features = ["brand", "fuel_type", "transmission", "accident_history"]

    X = df_prepared[numeric_features + categorical_features]
    y = df_prepared[target_col]

    # 3. Разделение выборки 80% / 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n2. Данные разделены на обучающую и тестовую выборки (80/20, random_state=42):")
    print(f"   - Размер обучающей выборки (X_train): {X_train.shape[0]} строк")
    print(f"   - Размер тестовой выборки   (X_test) : {X_test.shape[0]} строк")

    # 4. Демонстрация способов кодирования и масштабирования
    demonstrate_transformations(X_train)

    # 5. Построение ColumnTransformer
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    print("\n3. Сконструирован единый ColumnTransformer:")
    print("   - Числовые признаки -> SimpleImputer(median) + StandardScaler")
    print("   - Категориальные признаки -> SimpleImputer(most_frequent) + OneHotEncoder")

    return X_train, X_test, y_train, y_test, preprocessor, numeric_features, categorical_features


if __name__ == "__main__":
    from task1_initial_analysis import run_task1
    df = run_task1()
    run_task3(df)
