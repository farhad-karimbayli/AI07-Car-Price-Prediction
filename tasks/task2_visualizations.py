"""
Задание 2: Визуализация данных
Постройте:
• Гистограмму распределения стоимости.
• Диаграмму зависимости стоимости от пробега.
• Диаграмму зависимости стоимости от возраста.
• Столбчатую диаграмму средней стоимости по маркам.
Добавьте названия графиков и подписи осей. По каждому графику запишите краткий вывод.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def run_task2(df: pd.DataFrame, output_dir: str = "output", show: bool = False):
    print("=" * 65)
    print("ЗАДАНИЕ 2: Визуализация данных и формулирование выводов")
    print("=" * 65)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Гистограмма распределения стоимости
    # -------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.hist(df["price_azn"], bins=25, color="#3a86ff", edgecolor="black", alpha=0.8)
    ax1.axvline(df["price_azn"].mean(), color="red", linestyle="--", linewidth=2, label=f"Среднее: {df['price_azn'].mean():.0f} AZN")
    ax1.axvline(df["price_azn"].median(), color="orange", linestyle="-", linewidth=2, label=f"Медиана: {df['price_azn'].median():.0f} AZN")
    ax1.set_title("Распределение стоимости автомобилей (price_azn)", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Стоимость (AZN)", fontsize=12, labelpad=10)
    ax1.set_ylabel("Количество автомобилей (Frequency)", fontsize=12, labelpad=10)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    f1_file = out_path / "plot1_price_distribution.png"
    fig1.savefig(f1_file, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig1)

    print("\n1. Гистограмма распределения стоимости:")
    print(f"   График сохранен: {f1_file}")
    print("   ВЫВОД: Распределение цен близко к колоколообразному с легкой правосторонней")
    print("   асимметрией. Основная масса подержанных автомобилей находится в диапазоне")
    print("   от 15,000 до 35,000 AZN. Среднее значение (~27,000 AZN) близко к медиане.")

    # -------------------------------------------------------------
    # 2. Диаграмма зависимости стоимости от пробега
    # -------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    # Исключаем пропуски пробега для графика
    clean_mileage = df.dropna(subset=["mileage_km"])
    ax2.scatter(clean_mileage["mileage_km"], clean_mileage["price_azn"], color="#e63946", alpha=0.7, edgecolors="black", linewidths=0.5)
    ax2.set_title("Зависимость стоимости автомобиля от пробега", fontsize=14, fontweight="bold", pad=15)
    ax2.set_xlabel("Пробег (км)", fontsize=12, labelpad=10)
    ax2.set_ylabel("Стоимость (AZN)", fontsize=12, labelpad=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    f2_file = out_path / "plot2_price_vs_mileage.png"
    fig2.savefig(f2_file, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig2)

    print("\n2. Диаграмма зависимости стоимости от пробега:")
    print(f"   График сохранен: {f2_file}")
    print("   ВЫВОД: Наблюдается выраженная отрицательная корреляция: с увеличением пробега")
    print("   цена автомобиля закономерно снижается. Автомобили с пробегом свыше 200,000 км")
    print("   стоят существенно дешевле новых машин с минимальным пробегом.")

    # -------------------------------------------------------------
    # 3. Диаграмма зависимости стоимости от возраста
    # -------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(df["age_years"], df["price_azn"], color="#2a9d8f", alpha=0.75, s=60, edgecolors="black", linewidths=0.5)
    # Линия тренда по средним ценам каждого года
    mean_by_age = df.groupby("age_years")["price_azn"].mean()
    ax3.plot(mean_by_age.index, mean_by_age.values, color="#e76f51", linestyle="-", linewidth=2.5, marker="o", label="Средняя цена по годам")
    ax3.set_title("Зависимость стоимости автомобиля от возраста", fontsize=14, fontweight="bold", pad=15)
    ax3.set_xlabel("Возраст автомобиля (лет)", fontsize=12, labelpad=10)
    ax3.set_ylabel("Стоимость (AZN)", fontsize=12, labelpad=10)
    ax3.legend(fontsize=11)
    ax3.grid(True, linestyle=":", alpha=0.6)
    f3_file = out_path / "plot3_price_vs_age.png"
    fig3.savefig(f3_file, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig3)

    print("\n3. Диаграмма зависимости стоимости от возраста:")
    print(f"   График сохранен: {f3_file}")
    print("   ВЫВОД: Наблюдается сильная обратная линейная зависимость: каждый год эксплуатации")
    print("   снижает остаточную стоимость машины. Наибольшая цена у авто в возрасте 1-3 лет")
    print("   (свыше 35,000-45,000 AZN), тогда как к 12-15 годам цена падает до 5,000-20,000 AZN.")

    # -------------------------------------------------------------
    # 4. Столбчатая диаграмма средней стоимости по маркам
    # -------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    brand_means = df.groupby("brand")["price_azn"].mean().sort_values(ascending=False)
    colors = ["#4361ee", "#4cc9f0", "#7209b7", "#f72585"]
    bars = ax4.bar(brand_means.index, brand_means.values, color=colors, edgecolor="black", width=0.55)
    ax4.set_title("Средняя стоимость автомобилей по маркам (брендам)", fontsize=14, fontweight="bold", pad=15)
    ax4.set_xlabel("Марка автомобиля", fontsize=12, labelpad=10)
    ax4.set_ylabel("Средняя стоимость (AZN)", fontsize=12, labelpad=10)
    ax4.grid(axis="y", linestyle=":", alpha=0.6)
    ax4.set_ylim(0, brand_means.max() * 1.18)

    # Подписи значений над столбцами
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, height + 800, f"{height:,.0f} AZN", ha="center", va="bottom", fontsize=11, fontweight="bold")

    f4_file = out_path / "plot4_mean_price_by_brand.png"
    fig4.savefig(f4_file, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig4)

    print("\n4. Столбчатая диаграмма средней стоимости по маркам:")
    print(f"   График сохранен: {f4_file}")
    print("   ВЫВОД: Премиальный бренд BMW демонстрирует максимальную среднюю стоимость")
    print("   (свыше 34,000 AZN), следом идут Toyota, Hyundai и Kia (~24,000 - 27,000 AZN).")
    print("   Бренд является ключевым ценообразующим категориальным фактором.")


if __name__ == "__main__":
    from task1_initial_analysis import run_task1
    df = run_task1()
    run_task2(df, show=False)
