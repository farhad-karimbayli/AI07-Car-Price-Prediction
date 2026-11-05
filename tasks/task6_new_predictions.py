"""
Задание 6: Прогноз стоимости для новых автомобилей
Получите прогноз стоимости для следующих автомобилей:
1. Возраст: 5,  Пробег: 75 000,  Объём: 1.6, Мощность: 110, Марка: Toyota,  Топливо: petrol, Коробка: automatic, ДТП: no
2. Возраст: 10, Пробег: 180 000, Объём: 2.0, Мощность: 140, Марка: Hyundai, Топливо: diesel, Коробка: manual,    ДТП: yes
3. Возраст: 3,  Пробег: 40 000,  Объём: 2.5, Мощность: 170, Марка: BMW,     Топливо: petrol, Коробка: automatic, ДТП: no

Для новых данных выполните те же преобразования, что и при обучении.
"""

import pandas as pd
from sklearn.pipeline import Pipeline


def run_task6(full_pipeline: Pipeline):
    print("=" * 65)
    print("ЗАДАНИЕ 6: Прогнозирование стоимости для новых автомобилей")
    print("=" * 65)

    # 1. Формирование датафрейма с новыми автомобилями по условию задачи
    new_cars = pd.DataFrame([
        {
            "age_years": 5,
            "mileage_km": 75000,
            "engine_l": 1.6,
            "power_hp": 110,
            "brand": "Toyota",
            "fuel_type": "petrol",
            "transmission": "automatic",
            "accident_history": "no",
        },
        {
            "age_years": 10,
            "mileage_km": 180000,
            "engine_l": 2.0,
            "power_hp": 140,
            "brand": "Hyundai",
            "fuel_type": "diesel",
            "transmission": "manual",
            "accident_history": "yes",
        },
        {
            "age_years": 3,
            "mileage_km": 40000,
            "engine_l": 2.5,
            "power_hp": 170,
            "brand": "BMW",
            "fuel_type": "petrol",
            "transmission": "automatic",
            "accident_history": "no",
        },
    ])

    # 2. Вычисление engineered признака mileage_per_year
    new_cars["mileage_per_year"] = new_cars["mileage_km"] / new_cars["age_years"]

    # 3. Вычисление прогноза через обученный сквозной Pipeline
    predictions = full_pipeline.predict(new_cars)
    new_cars["predicted_price_azn"] = predictions

    print("\nХарактеристики автомобилей и полученные прогнозы стоимости:")
    cols_display = [
        "brand", "age_years", "mileage_km", "engine_l", "power_hp",
        "fuel_type", "transmission", "accident_history", "predicted_price_azn"
    ]
    formatted_table = new_cars[cols_display].copy()
    formatted_table["predicted_price_azn"] = formatted_table["predicted_price_azn"].apply(lambda v: f"{v:,.2f} AZN")

    print(formatted_table.to_string(index=True))

    print("\nДетальные результаты по каждому автомобилю:")
    for i, (_, row) in enumerate(new_cars.iterrows(), 1):
        print(f"\n   Автомобиль #{i}:")
        print(f"   • Марка: {row['brand']}, Возраст: {row['age_years']} лет, Пробег: {row['mileage_km']:,} км")
        print(f"   • Двигатель: {row['engine_l']} л ({row['power_hp']} л.с.), Топливо: {row['fuel_type']}, КПП: {row['transmission']}, ДТП: {row['accident_history']}")
        print(f"   --> ПРОГНОЗНАЯ СТОИМОСТЬ: {row['predicted_price_azn']:,.2f} AZN")

    return new_cars


if __name__ == "__main__":
    from task1_initial_analysis import run_task1
    from task3_preprocessing import run_task3
    from task4_model_evaluation import run_task4

    df = run_task1()
    X_train, X_test, y_train, y_test, preprocessor, _, _ = run_task3(df)
    pipeline, _, _ = run_task4(X_train, X_test, y_train, y_test, preprocessor)
    run_task6(pipeline)
