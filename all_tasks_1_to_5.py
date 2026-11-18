import timeit
from decimal import Decimal
from pathlib import Path
import numpy as np

# ============================================================
# ЗАДАНИЕ 1. ЧИСЛА С ПЛАВАЮЩЕЙ ТОЧКОЙ
# ============================================================

# 1.1 Массив float32, сумма и сравнение с точным 0.6
print("\n--- 1.1 Сумма float32 ---")
arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
s = np.sum(arr, dtype=np.float32)
print("Массив:", arr, "| тип:", arr.dtype)
print("Сумма:", s, "| точное значение:", Decimal("0.6"))
print("Отклонение:", Decimal(str(float(s))) - Decimal("0.6"))

# 1.2 Машинное эпсилон
print("\n--- 1.2 Машинное эпсилон ---")
print("float32:", np.finfo(np.float32).eps)
print("float64:", np.finfo(np.float64).eps)

# 1.3 Катастрофическая отмена
print("\n--- 1.3 Катастрофическая отмена ---")
a32, b32 = np.float32(100_000_001), np.float32(100_000_000)
print("float32:", a32, "-", b32, "=", a32 - b32,
      "(точная разность целых чисел = 1)")
a64 = np.float64(10_000_000_000_000_001)
b64 = np.float64(10_000_000_000_000_000)
print("float64:", a64, "-", b64, "=", a64 - b64,
      "(точная разность целых чисел = 1)")


# ============================================================
# ЗАДАНИЕ 2. ИНДЕКСАЦИЯ И BROADCASTING
# ============================================================

# 2.1 Массив A формы (4, 5), строка, столбец, подмассив
print("\n--- 2.1 Индексация и срезы ---")
A = np.arange(1, 21).reshape(4, 5)
print("A:\n", A)
print("Вторая строка:", A[1, :])
print("Столбец с индексом 3:", A[:, 3])
print("Строки 1-3 и столбцы 2-4:\n", A[1:4, 2:5])

# 2.2 Fancy-индексация: строки [0, 2, 3], столбцы [1, 4]
print("\n--- 2.2 Fancy-индексация ---")
print(A[np.ix_([0, 2, 3], [1, 4])])

# 2.3 Логическая индексация: заменить отрицательные значения на 0
print("\n--- 2.3 Логическая индексация ---")
A2 = np.array([[1, -2, 3, -4, 5], [-6, 7, -8, 9, -10],
               [11, -12, 13, -14, 15], [-16, 17, -18, 19, -20]])
print("До:\n", A2)
A2[A2 < 0] = 0
print("После:\n", A2)

# 2.4 Broadcasting: B (4, 1) + C (1, 5)
print("\n--- 2.4 Broadcasting ---")
B = np.array([[10], [20], [30], [40]])
C = np.array([[1, 2, 3, 4, 5]])
print("B, форма", B.shape, ":\n", B)
print("C, форма", C.shape, ":\n", C)
print("B + C, форма", (B + C).shape, ":\n", B + C)


# ============================================================
# ЗАДАНИЕ 3. MEMORY-MAPPED ФАЙЛ
# ============================================================

# 3.1 Файл на диске, форма (1000, 1000), float64, случайные числа
print("\n--- 3.1 Memory-mapped файл ---")
memmap_path = Path(__file__).with_name("array_memmap.dat")
mapped = np.memmap(memmap_path, dtype=np.float64, mode="w+", shape=(1000, 1000))
rng = np.random.default_rng(42)
mapped[:] = rng.random((1000, 1000))
mapped.flush()
print("Файл:", memmap_path, "| форма:", mapped.shape, "| тип:", mapped.dtype)

# 3.2 Чтение строки и столбца без загрузки всего массива в обычный ndarray
print("\n--- 3.2 Чтение строки и столбца ---")
print("Столбец 3, первые 10 элементов:", mapped[:, 3][:10])
print("Строка 10, первые 10 элементов:", mapped[10, :10])

# 3.3 Обход по строкам и столбцам, измерение через timeit
print("\n--- 3.3 Row-major и column-major ---")
def traverse_rows(x):
    total = 0.0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            total += x[i, j]
    return total

def traverse_columns(x):
    total = 0.0
    for j in range(x.shape[1]):
        for i in range(x.shape[0]):
            total += x[i, j]
    return total

t_rows = timeit.repeat(lambda: traverse_rows(mapped), number=1, repeat=3)
t_cols = timeit.repeat(lambda: traverse_columns(mapped), number=1, repeat=3)
print("Row-major, секунды:", t_rows, "| лучшее:", min(t_rows))
print("Column-major, секунды:", t_cols, "| лучшее:", min(t_cols))
print("Обычно row-major быстрее: в C-order соседние элементы строки лежат рядом в памяти.")


# ============================================================
# ЗАДАНИЕ 4. РАЗРЕЖЕННЫЕ МАТРИЦЫ
# ============================================================

try:
    from scipy import sparse
except ImportError:
    raise SystemExit("Для задания 4 установите SciPy: pip install scipy")

# 4.1 Разреженная матрица 5x5, плотность 20%, формат CSR
print("\n--- 4.1 Создание CSR ---")
dense_small = np.zeros((5, 5), dtype=np.float64)
for value, (i, j) in enumerate([(0, 0), (1, 3), (2, 1), (3, 4), (4, 2)], 1):
    dense_small[i, j] = value
csr = sparse.csr_matrix(dense_small)
print("Исходная матрица:\n", dense_small)
print("CSR:\n", csr)
print("Плотность:", csr.nnz / 25)

# 4.2 Преобразование в COO и CSC
print("\n--- 4.2 Преобразование в COO и CSC ---")
coo = csr.tocoo()
csc = csr.tocsc()
print("Формат COO:", coo.getformat())
print("Формат CSC:", csc.getformat())

# 4.3 Умножение разреженной матрицы на плотный вектор
print("\n--- 4.3 Умножение на вектор ---")
vector = np.array([1, 2, 3, 4, 5], dtype=np.float64)
print("Вектор:", vector)
print("Результат CSR @ vector:", csr @ vector)

# 4.4 Сравнение умножения sparse и dense матриц 1000x1000, density=0.01
print("\n--- 4.4 Сравнение sparse и dense ---")
n, density = 1000, 0.01
large_sparse = sparse.random(n, n, density=density, format="csr",
                             dtype=np.float64, random_state=42)
large_dense = large_sparse.toarray()
v = rng.random(n)
sparse_time = timeit.repeat(lambda: large_sparse @ v, number=20, repeat=3)
dense_time = timeit.repeat(lambda: large_dense @ v, number=20, repeat=3)
print("Sparse, 20 умножений:", sparse_time, "| лучшее:", min(sparse_time))
print("Dense, 20 умножений:", dense_time, "| лучшее:", min(dense_time))
print("Время зависит от компьютера; при низкой плотности sparse обычно экономит память.")


# ============================================================
# ЗАДАНИЕ 5. УСКОРЕНИЕ ЦИКЛОВ С ПОМОЩЬЮ CYTHON
# ============================================================

# 5.1 Python-функция суммы квадратов
print("\n--- 5.1 Python-функция ---")
def sum_squares_python(values):
    total = 0.0
    for value in values:
        total += value * value
    return total

print("Для [1, 2, 3, 4]:", sum_squares_python([1.0, 2.0, 3.0, 4.0]))

# 5.2 Cython-функция находится в отдельном обязательном файле
# sum_squares_cython.pyx. pyximport скомпилирует её при импорте.
print("\n--- 5.2 Подключение Cython-функции ---")
try:
    import pyximport
    pyximport.install(language_level=3, inplace=False)
    from sum_squares_cython import sum_squares_cython
    print("Cython-модуль скомпилирован.")
    print("Для [1, 2, 3, 4]:",
          sum_squares_cython(np.array([1., 2., 3., 4.], dtype=np.float64)))
except Exception as error:
    sum_squares_cython = None
    print("Cython не подключён:", error)
    print("Установите: pip install cython setuptools")
    print("Проверьте, что sum_squares_cython.pyx лежит рядом с этим файлом.")

# 5.3 Сравнение на массиве из 10^7 элементов
print("\n--- 5.3 Сравнение времени на 10^7 элементов ---")
values = rng.random(10**7)
python_times = timeit.repeat(lambda: sum_squares_python(values), number=1, repeat=3)
print("Python, секунды:", python_times, "| лучшее:", min(python_times))
if sum_squares_cython is not None:
    cython_times = timeit.repeat(lambda: sum_squares_cython(values), number=1, repeat=3)
    print("Cython, секунды:", cython_times, "| лучшее:", min(cython_times))
    print("Ускорение примерно:", min(python_times) / min(cython_times), "раз")
else:
    print("Сравнение с Cython пропущено: установите Cython и запустите файл снова.")

print("\nВсе задания обработаны.")
