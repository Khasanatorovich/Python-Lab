"""
Установка библиотек
    pip install numpy pandas scipy
Чтобы пропустить самые тяжёлые вычисления
    python all_tasks.py --skip-heavy
"""

import sys
import os
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


# ---------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------

def заголовок(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def пропустить_тяжёлые():
    return "--skip-heavy" in sys.argv


# =====================================================================
# ТЕМА 1.1 — ЗАДАНИЯ 6–8
# =====================================================================

def task_6():
    заголовок("ТЕМА 1.1 — ЗАДАНИЕ 6: индексация NumPy и broadcasting")

    # 6.1
    print("\n6.1 — Срезы массива")
    arr = np.arange(20)
    print("Исходный массив:", arr)
    print("Каждый третий элемент, начиная с индекса 1:", arr[1::3])
    print("Элементы с 5-го по 15-й (15 не включается), в обратном порядке:",
          arr[14:4:-1])
    print("Последние 5 элементов в обратном порядке:", arr[-1:-6:-1])

    # 6.2
    print("\n6.2 — Индексация матрицы 6x6")
    matrix = np.random.randint(0, 10, size=(6, 6))
    print("Исходная матрица:\n", matrix)
    print("Подматрица: строки 2–4 и столбцы 1–3 (включительно):\n",
          matrix[2:5, 1:4])
    print("Главная диагональ:", np.diagonal(matrix))
    rows, cols = np.indices(matrix.shape)
    print("Шахматный узор (чётная сумма индексов):\n",
          matrix[(rows + cols) % 2 == 0].reshape(-1))

    # 6.3
    print("\n6.3 — Индексация массивов и матриц")
    a = np.array([10, 20, 30, 40, 50, 60])
    m = np.arange(12).reshape(3, 4)
    print("a по индексам [0, 2, 4]:", a[[0, 2, 4]])
    print("Матрица m:\n", m)
    print("Элементы (0,1), (1,3), (2,0):", m[[0, 1, 2], [1, 3, 0]])
    print("Блок строк [0, 2] и столбцов [1, 3]:\n",
          m[np.ix_([0, 2], [1, 3])])

    # 6.4
    print("\n6.4 — Булева индексация и смена знака")
    x = np.random.randn(1000)
    print("Количество элементов больше 1:", np.count_nonzero(x > 1))

    x_without_negatives = x.copy()
    x_without_negatives[x_without_negatives < 0] = 0
    print("Количество отрицательных после замены:",
          np.count_nonzero(x_without_negatives < 0))

    # Обнуляем x[i], если его соседи x[i-1] и x[i+1] имеют разные знаки.
    sign_change = (x[:-2] * x[2:]) < 0
    x_neighbors_zeroed = x.copy()
    x_neighbors_zeroed[1:-1][sign_change] = 0
    print("Количество обнулённых элементов между соседями разных знаков:",
          np.count_nonzero(sign_change))


def task_7():
    заголовок("ТЕМА 1.1 — ЗАДАНИЕ 7: большие массивы и производительность")

    if пропустить_тяжёлые():
        print("Задание пропущено из-за параметра --skip-heavy.")
        return

    # 7.1
    print("\n7.1 — Создание файла big.dat")
    shape = (50000, 200)
    filename = Path("big.dat")
    mm = np.memmap(filename, dtype=np.float32, mode="w+", shape=shape)
    rng = np.random.default_rng(42)
    # Заполняем блоками, чтобы не создавать второй массив размером со весь файл.
    for start in range(0, shape[0], 1000):
        end = min(start + 1000, shape[0])
        mm[start:end] = rng.random((end - start, shape[1]), dtype=np.float32)
    mm.flush()
    del mm
    expected_size = shape[0] * shape[1] * np.dtype(np.float32).itemsize
    actual_size = filename.stat().st_size
    print(f"Размер файла: {actual_size:,} байт")
    print(f"Ожидаемый размер: {expected_size:,} байт")
    print("Размер совпадает:", actual_size == expected_size)

    # 7.2
    print("\n7.2 — Min, max и среднее по memmap батчами")
    data = np.memmap(filename, dtype=np.float32, mode="r", shape=shape)
    global_min = np.inf
    global_max = -np.inf
    total_sum = 0.0
    total_count = 0
    batch_size = 1000
    for start in range(0, shape[0], batch_size):
        batch = data[start:start + batch_size]
        global_min = min(global_min, float(batch.min()))
        global_max = max(global_max, float(batch.max()))
        total_sum += float(batch.sum(dtype=np.float64))
        total_count += batch.size
    print("Глобальный min:", global_min)
    print("Глобальный max:", global_max)
    print("Глобальное среднее:", total_sum / total_count)
    del data

    # 7.3
    print("\n7.3 — Сумма C-order и F-order")
    n = 3000
    c_array = np.random.random((n, n))  # C-order по умолчанию
    f_array = np.asfortranarray(c_array)

    def row_sum_python(array):
        result = np.zeros(array.shape[0], dtype=np.float64)
        for i in range(array.shape[0]):
            s = 0.0
            for j in range(array.shape[1]):
                s += array[i, j]
            result[i] = s
        return result

    def col_sum_python(array):
        result = np.zeros(array.shape[1], dtype=np.float64)
        for j in range(array.shape[1]):
            s = 0.0
            for i in range(array.shape[0]):
                s += array[i, j]
            result[j] = s
        return result

    for name, array in (("C-order", c_array), ("F-order", f_array)):
        start = time.perf_counter()
        row_result = row_sum_python(array)
        row_time = time.perf_counter() - start

        start = time.perf_counter()
        col_result = col_sum_python(array)
        col_time = time.perf_counter() - start

        print(f"{name}: построчно {row_time:.3f} сек.; "
              f"постолбцово {col_time:.3f} сек.")
        print("Проверка суммы:", np.isclose(row_result.sum(), col_result.sum()))

    # 7.4
    print("\n7.4 — Сравнение трёх способов вычисления на 10 млн элементов")
    count = 10_000_000
    rng = np.random.default_rng(123)
    a_np = rng.random(count)
    b_np = rng.random(count)

    # NumPy в одну строку
    start = time.perf_counter()
    result_np = (a_np + b_np) / (a_np - b_np + 1)
    numpy_time = time.perf_counter() - start
    print(f"NumPy в одну строку: {numpy_time:.3f} сек.")

    # NumPy с out= для переиспользования буфера
    out = np.empty_like(a_np)
    temp = np.empty_like(a_np)
    start = time.perf_counter()
    np.add(a_np, b_np, out=out)
    np.subtract(a_np, b_np, out=temp)
    np.add(temp, 1, out=temp)
    np.divide(out, temp, out=out)
    out_time = time.perf_counter() - start
    print(f"NumPy с out=: {out_time:.3f} сек.")

    # Python list. Конвертация и вычисление входят в измеряемое время.
    # Списки создаются только на время этого теста, затем удаляются.
    start = time.perf_counter()
    a_list = a_np.tolist()
    b_list = b_np.tolist()
    result_list = [(x + y) / (x - y + 1) for x, y in zip(a_list, b_list)]
    list_time = time.perf_counter() - start
    print(f"Python-цикл/list comprehension: {list_time:.3f} сек.")

    del a_list, b_list, result_list, result_np, out, temp, a_np, b_np
    gc.collect()


def task_8():
    заголовок("ТЕМА 1.1 — ЗАДАНИЕ 8: разреженные матрицы")

    if пропустить_тяжёлые():
        print("Задание пропущено из-за параметра --skip-heavy.")
        return

    # 8.1
    print("\n8.1 — COO -> CSR")
    rng = np.random.default_rng(42)
    size = 5000
    nnz_input = 10_000
    row = rng.integers(0, size, size=nnz_input)
    col = rng.integers(0, size, size=nnz_input)
    values = rng.random(nnz_input)
    coo = sparse.coo_matrix((values, (row, col)), shape=(size, size))
    csr = coo.tocsr()
    print("Размер матрицы:", csr.shape)
    print("Число переданных элементов:", nnz_input)
    print("nnz после преобразования (совпадающие координаты объединены):", csr.nnz)
    print("Проверка nnz:", csr.nnz <= nnz_input)

    # 8.2
    print("\n8.2 — Сравнение dense и CSR")
    n = 4000
    density = 0.005
    dense = sparse.random(
        n, n, density=density, format="csr",
        dtype=np.float64, random_state=42
    ).toarray()
    csr_matrix = sparse.csr_matrix(dense)
    vector = rng.random(n)

    dense_bytes = dense.nbytes
    csr_bytes = (csr_matrix.data.nbytes + csr_matrix.indices.nbytes
                 + csr_matrix.indptr.nbytes)
    print(f"Dense память: {dense_bytes:,} байт")
    print(f"CSR память:   {csr_bytes:,} байт")

    start = time.perf_counter()
    dense_result = dense @ vector
    dense_time = time.perf_counter() - start

    start = time.perf_counter()
    sparse_result = csr_matrix @ vector
    sparse_time = time.perf_counter() - start

    print(f"Dense умножение: {dense_time:.6f} сек.")
    print(f"CSR умножение:   {sparse_time:.6f} сек.")
    print("Результаты совпадают:", np.allclose(dense_result, sparse_result))

    # 8.3
    print("\n8.3 — Время умножения в зависимости от плотности")
    n = 2000
    densities = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    vector = rng.random(n)
    results = []

    for density in densities:
        matrix = sparse.random(
            n, n, density=density, format="csr",
            dtype=np.float64, random_state=42
        )
        start = time.perf_counter()
        _ = matrix @ vector
        elapsed = time.perf_counter() - start
        results.append((density, elapsed))
        print(f"Плотность {density:>5}: {elapsed:.6f} сек.")

    # Для сравнения — плотное умножение при каждой плотности.
    print("\nВремя dense для матрицы 2000x2000:")
    dense_matrix = np.zeros((n, n), dtype=np.float64)
    for density in densities:
        # Используем ту же плотность для создания тестовой dense-матрицы.
        dense_matrix.fill(0)
        nnz = int(n * n * density)
        indices = rng.choice(n * n, size=nnz, replace=False)
        dense_matrix.flat[indices] = rng.random(nnz)
        start = time.perf_counter()
        _ = dense_matrix @ vector
        elapsed = time.perf_counter() - start
        print(f"Плотность {density:>5}: {elapsed:.6f} сек.")

    print("\nТочка перехода зависит от компьютера и библиотек; "
          "сравните времена CSR и dense в выводе.")


# =====================================================================
# ТЕМА 2 — ЗАДАНИЯ 1–6 (Pandas)
# =====================================================================

def prepare_dataframe():
    """Демонстрационные данные: замените их своими, если они есть."""
    employees = {
        "name": [
            "Анна", "Борис", "Виктор", "Галина", "Дмитрий",
            "Елена", "Игорь", "Мария", "Олег", "Светлана",
            "Павел", "Наталья"
        ],
        "department": [
            "IT", "IT", "HR", "HR", "Sales", "Sales",
            "IT", "Finance", "Finance", "IT", "Sales", "HR"
        ],
        "salary": [
            90000, 75000, 60000, np.nan, 82000, 71000,
            np.nan, 95000, 88000, 105000, 68000, 64000
        ],
        "bonus": [
            10000, np.nan, 5000, 4000, 8000, np.nan,
            12000, 15000, 9000, 20000, 3000, 4500
        ],
        "hire_date": pd.to_datetime([
            "2021-03-15", "2022-07-01", "2020-01-20", "2023-02-10",
            "2019-11-05", "2022-09-12", "2021-06-30", "2018-04-02",
            "2020-08-19", "2017-12-01", "2023-05-17", "2022-01-25"
        ])
    }
    return pd.DataFrame(employees)


def task_1(df):
    заголовок("ТЕМА 2 — ЗАДАНИЕ 1: Series, DataFrame и первичная информация")

    # 1.1
    print("\n1.1 — Series из списка с произвольным индексом")
    series = pd.Series(
        [10, 20, 30, 40, 50],
        index=["a", "b", "c", "d", "e"],
        name="numbers"
    )
    print(series)

    # 1.2
    print("\n1.2 — DataFrame из словаря")
    data = {
        "name": ["Иван", "Мария", "Пётр"],
        "age": [25, 30, 28],
        "city": ["Москва", "Казань", "Тула"]
    }
    example_df = pd.DataFrame(data)
    print(example_df)

    # 1.3
    print("\n1.3 — Первые/последние строки, типы и статистика")
    print("Первые строки:\n", df.head())
    print("Последние строки:\n", df.tail())
    print("Типы данных:\n", df.dtypes)
    print("Статистика:\n", df.describe(include="all"))


def task_2(df):
    заголовок("ТЕМА 2 — ЗАДАНИЕ 2: выборка, фильтрация и новые столбцы")

    # 2.1
    print("\n2.1 — name и salary для сотрудников IT")
    print(df.loc[df["department"] == "IT", ["name", "salary"]])

    # 2.2
    print("\n2.2 — Первые 3 строки и первые 2 столбца через iloc")
    print(df.iloc[:3, :2])

    # 2.3
    print("\n2.3 — Сотрудники с зарплатой выше 70000")
    print(df.loc[df["salary"] > 70000])

    # 2.4
    print("\n2.4 — total_comp = salary + bonus (пропуски bonus = 0)")
    df["total_comp"] = df["salary"] + df["bonus"].fillna(0)
    print(df[["name", "salary", "bonus", "total_comp"]])

    # 2.5
    print("\n2.5 — Удаление hire_date")
    df = df.drop(columns=["hire_date"])
    print(df.head())


def task_3(df):
    заголовок("ТЕМА 2 — ЗАДАНИЕ 3: пропуски")

    # 3.1
    print("\n3.1 — Количество пропусков в каждом столбце")
    print(df.isna().sum())

    # 3.2
    print("\n3.2 — Заполнение salary средним значением по отделу")
    df["salary"] = df.groupby("department")["salary"].transform(
        lambda values: values.fillna(values.mean())
    )
    print(df[["name", "department", "salary"]])

    # 3.3
    print("\n3.3 — Заполнение bonus нулём")
    df["bonus"] = df["bonus"].fillna(0)
    df["total_comp"] = df["salary"] + df["bonus"]
    print(df[["name", "bonus", "total_comp"]])

    # 3.4
    print("\n3.4 — Удаление строк, где salary всё ещё пропущен")
    before = len(df)
    df = df.dropna(subset=["salary"])
    print("Удалено строк:", before - len(df))
    print("Осталось пропусков salary:", df["salary"].isna().sum())


def task_4(df):
    заголовок("ТЕМА 2 — ЗАДАНИЕ 4: groupby и сводные таблицы")

    # 4.1
    print("\n4.1 — Средняя зарплата по отделам")
    print(df.groupby("department")["salary"].mean())

    # 4.2
    print("\n4.2 — Средняя зарплата, максимальный бонус, число сотрудников")
    grouped = df.groupby("department").agg(
        average_salary=("salary", "mean"),
        max_bonus=("bonus", "max"),
        employees_count=("name", "count")
    )
    print(grouped)

    # 4.3
    print("\n4.3 — Сводная таблица: индекс — отдел, значение — средняя зарплата")
    pivot = pd.pivot_table(
        df, index="department", values="salary", aggfunc="mean"
    )
    print(pivot)

    # 4.4
    print("\n4.4 — Сортировка отделов по средней зарплате по убыванию")
    print(grouped.sort_values("average_salary", ascending=False))


def task_5(df):
    заголовок("ТЕМА 2 — ЗАДАНИЕ 5: merge и concat")

    # 5.1
    print("\n5.1 — DataFrame managers с отделами и руководителями")
    managers = pd.DataFrame({
        "department": ["IT", "HR", "Sales", "Finance"],
        "manager": ["Алексей", "Ольга", "Сергей", "Ирина"]
    })
    print(managers)

    # 5.2
    print("\n5.2 — LEFT JOIN сотрудников и managers по department")
    merged = df.merge(managers, on="department", how="left")
    print(merged)

    # 5.3
    print("\n5.3 — Добавление новой строки через concat")
    new_employee = pd.DataFrame([{
        "name": "Кирилл",
        "department": "IT",
        "salary": 78000,
        "bonus": 6000,
        "total_comp": 84000
    }])
    df = pd.concat([df, new_employee], ignore_index=True)
    print(df.tail())

    # 5.4
    print("\n5.4 — Сброс индекса после объединения")
    df = df.reset_index(drop=True)
    print(df)
    return df


def task_6_pandas(df):
    заголовок("ТЕМА 2 — ЗАДАНИЕ 6: ответы на вопросы по данным")

    # 6.1
    print("\n6.1 — Топ-3 сотрудников по total_comp")
    print(df.nlargest(3, "total_comp")[["name", "department", "total_comp"]])

    # 6.2
    print("\n6.2 — Средняя зарплата и средний бонус по отделам")
    averages = df.groupby("department").agg(
        average_salary=("salary", "mean"),
        average_bonus=("bonus", "mean")
    )
    print(averages)

    # 6.3
    print("\n6.3 — Доля сотрудников с зарплатой выше 75000 по отделам")
    share = df.assign(salary_above_75000=df["salary"] > 75000).groupby(
        "department"
    )["salary_above_75000"].mean()
    print(share)


def main():
    np.random.seed(42)

    # Сначала тема 1.1, задания 6–8
    task_6()
    task_7()
    task_8()

    # Затем тема 2, задания 1–6
    заголовок("ПОДГОТОВКА ДАННЫХ ДЛЯ PANDAS")
    df = prepare_dataframe()
    print("Исходная таблица сотрудников:\n", df)

    task_1(df)
    task_2(df)
    task_3(df)
    task_4(df)
    df = task_5(df)
    task_6_pandas(df)

    заголовок("РАБОТА ЗАВЕРШЕНА")


if __name__ == "__main__":
    main()
