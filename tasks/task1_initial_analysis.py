"""
Задание 1: Первичный анализ данных
• Загрузите данные и выведите первые 10 строк.
• Определите количество строк и столбцов, типы данных и количество пропусков.
• Выведите описательную статистику числовых столбцов.
• Найдите минимальную, максимальную и среднюю стоимость автомобиля.
"""

from pathlib import Path
import pandas as pd


def get_data_path() -> Path:
    candidates = [
        Path(__file__).parent.parent / "data" / "Used_Cars_Homework.csv",
        Path(__file__).parent.parent / "Used_Cars_Homework.csv",
        Path("Used_Cars_Homework.csv"),
        Path("data/Used_Cars_Homework.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def run_task1():
    print("=" * 65)
    print("ЗАДАНИЕ 1: Первичный анализ данных (Used_Cars_Homework.csv)")
    print("=" * 65)

    data_path = get_data_path()
    df = pd.read_csv(data_path)

    # 1. Первые 10 строк
    print("\n1. Первые 10 строк набора данных:")
    print(df.head(10))

    # 2. Размерность, типы данных и пропуски
    print("\n2. Информация о структуре данных:")
    print(f"   Количество строк: {df.shape[0]}")
    print(f"   Количество столбцов: {df.shape[1]}")

    print("\n   Типы данных по столбцам:")
    for col, dtype in df.dtypes.items():
        print(f"   - {col:18}: {dtype}")

    print("\n   Количество пропущенных значений (NaN):")
    missing = df.isna().sum()
    for col, count in missing.items():
        pct = (count / len(df)) * 100
        print(f"   - {col:18}: {count:3d} пропусков ({pct:4.1f}%)")

    # 3. Описательная статистика числовых признаков
    print("\n3. Описательная статистика числовых столбцов:")
    print(df.describe().round(2))

    # 4. Анализ целевой переменной (price_azn)
    min_price = df["price_azn"].min()
    max_price = df["price_azn"].max()
    mean_price = df["price_azn"].mean()
    median_price = df["price_azn"].median()

    print("\n4. Характеристики стоимости автомобиля (price_azn):")
    print(f"   - Минимальная стоимость : {min_price:,.2f} AZN")
    print(f"   - Максимальная стоимость: {max_price:,.2f} AZN")
    print(f"   - Средняя стоимость    : {mean_price:,.2f} AZN")
    print(f"   - Медианная стоимость   : {median_price:,.2f} AZN")

    return df


if __name__ == "__main__":
    run_task1()
