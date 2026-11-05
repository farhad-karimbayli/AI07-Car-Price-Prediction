"""
Задание 4: Обучение и оценка модели
• Обучите LinearRegression и получите прогнозы для тестовой выборки.
• Рассчитайте: MAE, MSE, RMSE, R².
• Объясните полученные результаты своими словами: что они говорят о качестве модели
  и в каких единицах выражены ошибки?
• Выведите таблицу для 10 тестовых автомобилей: реальная стоимость, прогноз и абсолютная ошибка.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_task4(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, preprocessor):
    print("=" * 65)
    print("ЗАДАНИЕ 4: Обучение модели LinearRegression и оценка качества")
    print("=" * 65)

    # 1. Сборка полного сквозного Pipeline (предобработка + модель)
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])

    # Обучение модели исключительно на обучающих данных (X_train, y_train)
    full_pipeline.fit(X_train, y_train)

    # Получение прогнозов для тестовой выборки
    y_pred = full_pipeline.predict(X_test)

    # 2. Расчет метрик качества
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n1. Метрики качества на тестовой выборке (20% данных):")
    print(f"   • MAE  (Mean Absolute Error)     : {mae:10.2f} AZN")
    print(f"   • MSE  (Mean Squared Error)      : {mse:10.2f} AZN²")
    print(f"   • RMSE (Root Mean Squared Error) : {rmse:10.2f} AZN")
    print(f"   • R²   (Коэффициент детерминации): {r2:10.4f} ({r2*100:.2f}%)")

    # 3. Объяснение результатов своими словами
    print("\n2. Интерпретация метрик и единицы измерения:")
    print("   • MAE (Средняя абсолютная ошибка): выражена в манатах (AZN). Показывает, что")
    print(f"     в среднем модель ошибается в оценке автомобиля на ~{mae:.0f} AZN в ту или иную сторону.")
    print("   • MSE (Среднеквадратичная ошибка): выражена в квадратах манатов (AZN²).")
    print("     Штрафует крупные ошибки сильнее за счет возведения разности в квадрат.")
    print("   • RMSE (Корень из MSE): выражен в манатах (AZN). Составляет ~{0:.0f} AZN,".format(rmse))
    print("     давая представление о величине стандартного отклонения ошибки прогноза.")
    print(f"   • R² (Коэффициент детерминации): безразмерная величина от -∞ до 1. Значение {r2:.3f}")
    print(f"     означает, что модель объясняет более {r2*100:.1f}% дисперсии (вариативности) стоимости")
    print("     автомобилей по имеющимся характеристикам. Это показатель очень высокого качества.")

    # 4. Таблица для 10 тестовых автомобилей
    test_comparison = pd.DataFrame({
        "Реальная цена (AZN)": y_test.values[:10],
        "Прогноз модели (AZN)": y_pred[:10],
        "Абсолютная ошибка (AZN)": np.abs(y_test.values[:10] - y_pred[:10]),
        "Относительная ошибка (%)": (np.abs(y_test.values[:10] - y_pred[:10]) / y_test.values[:10]) * 100,
    })

    print("\n3. Сравнительная таблица для первых 10 тестовых автомобилей:")
    print(test_comparison.round(2).to_string(index=True))

    return full_pipeline, y_pred, (mae, mse, rmse, r2)


if __name__ == "__main__":
    from task1_initial_analysis import run_task1
    from task3_preprocessing import run_task3

    df = run_task1()
    X_train, X_test, y_train, y_test, preprocessor, _, _ = run_task3(df)
    run_task4(X_train, X_test, y_train, y_test, preprocessor)
