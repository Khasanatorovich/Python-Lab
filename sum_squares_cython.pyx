# Файл: sum_squares_cython.pyx
# Cython-реализация суммы квадратов элементов массива.

cpdef double sum_squares_cython(double[:] values):
    cdef Py_ssize_t i
    cdef Py_ssize_t n = values.shape[0]
    cdef double total = 0.0

    for i in range(n):
        total += values[i] * values[i]

    return total
