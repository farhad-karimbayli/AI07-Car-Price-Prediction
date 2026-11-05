# AI 07 — Прогнозирование стоимости подержанного автомобиля

**Проект:** Multiple Linear Regression & Machine Learning Pipelines  
**Курс:** [AI-Applications-Using-Python-FSDE_Oct_24_3_ru](https://github.com/Zamanof/AI-Applications-Using-Python-FSDE_Oct_24_3_ru)  
**Датасет:** `Used_Cars_Homework.csv` (300 строк, 9 признаков)  
**Файл для сдачи:** `AI_07_Used_Cars_Price_Prediction.ipynb`

---

## 📁 Структура проекта `AI07`

```text
AI07/
├── data/
│   └── Used_Cars_Homework.csv      # Исходный набор данных
├── tasks/
│   ├── __init__.py
│   ├── task1_initial_analysis.py   # Задание 1: Первичный анализ, статистика, пропуски
│   ├── task2_visualizations.py     # Задание 2: 4 графика с выводами (цена, пробег, возраст, бренд)
│   ├── task3_preprocessing.py      # Задание 3: Feature Engineering, factorize/OHE, scalers, ColumnTransformer
│   ├── task4_model_evaluation.py   # Задание 4: Обучение LinearRegression, MAE/MSE/RMSE/R², таблица 10 авто
│   ├── task5_learning_curve.py     # Задание 5: 5-Fold Learning Curve, анализ переобучения
│   └── task6_new_predictions.py    # Задание 6: Прогноз для 3 новых автомобилей
├── output/
│   ├── plot1_price_distribution.png# Гистограмма распределения стоимости
│   ├── plot2_price_vs_mileage.png  # Scatter plot стоимости от пробега
│   ├── plot3_price_vs_age.png      # Scatter plot стоимости от возраста с линией тренда
│   ├── plot4_mean_price_by_brand.png # Столбчатая диаграмма средней цены по маркам
│   └── plot5_learning_curve.png    # Кривая обучения (Learning Curve) с доверительными интервалами
├── main.py                         # Главный исполняемый скрипт всего проекта
├── AI_07_Used_Cars_Price_Prediction.ipynb # Итоговый Jupyter Notebook с кодом, графиками и выводами
├── requirements.txt                # Зависимости проекта (pandas, numpy, matplotlib, scikit-learn)
├── README.md                       # Полное руководство, теория и результаты
└── .gitignore                      # Исключения (.venv, кэш)
```

---

## 🚀 Установка и запуск

### 1. Настройка виртуального окружения
С помощью `uv`:
```bash
cd AI07
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
Либо через стандартный `python3 -m venv`:
```bash
cd AI07
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Запуск программы
- **Запуск всех расчетов, генерации графиков и прогнозов:**
  ```bash
  python main.py
  ```
- **Запуск отдельного модуля (например, первичный анализ или графики):**
  ```bash
  python tasks/task1_initial_analysis.py
  python tasks/task2_visualizations.py
  python tasks/task4_model_evaluation.py
  ```
- **Интерактивный запуск блокнота:**
  Откройте `AI_07_Used_Cars_Price_Prediction.ipynb` в VS Code или Jupyter Notebook (ядро `.venv`).

---

## 📊 Результаты работы по разделам

### 1. Первичный анализ данных
- Размер выборки: **300 автомобилей**, 9 столбцов.
- Пропуски: `mileage_km` (17), `engine_l` (16), `fuel_type` (13), `transmission` (10).
- Статистика цен (`price_azn`):
  - Минимальная: **3,805.00 AZN**
  - Максимальная: **51,396.00 AZN**
  - Средняя: **27,806.57 AZN**
  - Медианная: **27,458.50 AZN**

### 2. Визуализация и выводы
1. **Гистограмма распределения цен:** нормальное унимодальное распределение с центром около 27 000 AZN.
2. **Стоимость от пробега:** выраженная отрицательная нелинейная корреляция: пробег свыше 200 тыс. км снижает цену машины в 2–3 раза.
3. **Стоимость от возраста:** сильнейший фактор амортизации: свежие автомобили (1–3 года) стоят 35–45 тыс. AZN, к 12–15 годам цена падает до 5–15 тыс. AZN.
4. **Средняя цена по брендам:** **BMW** (лидер премиум-сегмента, ~34,500 AZN), затем **Toyota** (~26,900 AZN), **Kia** (~25,200 AZN) и **Hyundai** (~24,800 AZN).

### 3. Предобработка данных (Feature Engineering & Pipeline)
- Создан признак среднегодового пробега: `mileage_per_year = mileage_km / age_years`.
- Разделение данных: **80% train / 20% test** (`random_state=42`).
- Сформирован `ColumnTransformer`:
  - **Числовые признаки:** `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`
  - **Категориальные признаки:** `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore')`

#### Сравнение методов кодирования и масштабирования:
- **`factorize` vs `One-Hot Encoding`:** `factorize` вводит фиктивный математический порядок ($0 < 1 < 2$), что вредно для линейных моделей. `One-Hot Encoding` создает независимые бинарные векторы для каждой марки.
- **`MinMaxScaler` vs `StandardScaler`:** `MinMaxScaler` сжимает данные в $[0, 1]$ и уязвим к выбросам; `StandardScaler` приводит к $\mu=0, \sigma=1$, устойчив к экстремальным значениям и оптимален для регрессии.

### 4. Метрики качества модели
- **MAE:** **1,530.82 AZN** (средняя ошибка оценки автомобиля всего ~1.5 тыс. манатов)
- **RMSE:** **1,944.59 AZN**
- **$R^2$:** **0.9572 (95.72%)** — модель объясняет более 95% дисперсии цен.

### 5. Анализ кривой обучения (Learning Curve)
- 5-кратная кросс-валидация на `X_train`.
- Оценка на валидации вырастает до **$R^2 \approx 0.947$**.
- Разрыв между Train и Validation $R^2$ минимален ($\approx 0.01$), дисперсия низкая $\rightarrow$ **переобучение отсутствует (Good Fit)**.

### 6. Прогнозы для новых автомобилей
| # | Марка | Возраст | Пробег | Объем | Мощность | Топливо | КПП | ДТП | Прогнозная стоимость |
| :-: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Toyota** | 5 лет | 75 000 км | 1.6 л | 110 л.с. | petrol | automatic | no | **31,170.63 AZN** |
| 2 | **Hyundai** | 10 лет | 180 000 км | 2.0 л | 140 л.с. | diesel | manual | yes | **15,912.01 AZN** |
| 3 | **BMW** | 3 года | 40 000 км | 2.5 л | 170 л.с. | petrol | automatic | no | **48,791.50 AZN** |
