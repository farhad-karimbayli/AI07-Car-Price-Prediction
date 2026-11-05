"""
AI Applications Using Python — AI 07 Car Price Prediction Homework
Главный исполняемый модуль проекта AI07.
"""

from tasks.task1_initial_analysis import run_task1
from tasks.task2_visualizations import run_task2
from tasks.task3_preprocessing import run_task3
from tasks.task4_model_evaluation import run_task4
from tasks.task5_learning_curve import run_task5
from tasks.task6_new_predictions import run_task6


def main():
    print("\n" + "#" * 70)
    print("#  КУРС: AI Applications Using Python")
    print("#  МОДУЛЬ: AI 07 — Машинное обучение: Регрессия и пайплайны")
    print("#  ПРОЕКТ: Прогнозирование стоимости подержанного автомобиля")
    print("#" * 70 + "\n")

    # 1. Первичный анализ данных
    df = run_task1()
    print("\n" + "-" * 65 + "\n")

    # 2. Визуализация и сохранение графиков
    run_task2(df, output_dir="output", show=False)
    print("\n" + "-" * 65 + "\n")

    # 3. Подготовка данных и демонстрация способов трансформации
    X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols = run_task3(df)
    print("\n" + "-" * 65 + "\n")

    # 4. Обучение модели и оценка метрик (MAE, MSE, RMSE, R2)
    pipeline, y_pred, metrics = run_task4(X_train, X_test, y_train, y_test, preprocessor)
    print("\n" + "-" * 65 + "\n")

    # 5. Проверка обучения: кривая обучения (Learning Curve)
    run_task5(pipeline, X_train, y_train, output_dir="output", show=False)
    print("\n" + "-" * 65 + "\n")

    # 6. Прогноз стоимости для новых автомобилей
    run_task6(pipeline)
    print("\n" + "#" * 70)
    print("#  Все задания проекта AI07 успешно выполнены!")
    print("#  Графики сохранены в директории output/")
    print("#  Итоговый блокнот: AI_07_Used_Cars_Price_Prediction.ipynb")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
