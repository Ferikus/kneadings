import numpy as np

DIM = 3
INFINITY = 100000

InfinityError = -0.2
KneadingDoNotEndError = -0.1


# Шаг РК-4, сохраняет значение в y_curr
def stepper_rk4(params, y_curr, dt):
    k1 = np.zeros(DIM)
    k2 = np.zeros(DIM)
    k3 = np.zeros(DIM)
    k4 = np.zeros(DIM)

    rhs(params, y_curr, k1)

    for i in range(DIM):
        k2[i] = y_curr[i] + k1[i] * dt / 2.0
    rhs(params, k2, k2)

    for i in range(DIM):
        k3[i] = y_curr[i] + k2[i] * dt / 2.0
    rhs(params, k3, k3)

    for i in range(DIM):
        k4[i] = y_curr[i] + k3[i] * dt
    rhs(params, k4, k4)

    for i in range(DIM):
        y_curr[i] += (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) * dt / 6.0


# Значение системы ДУ в точке
def rhs(params, y, dydt):
    a, b = params
    dydt[0] = y[1]
    dydt[1] = y[2]
    dydt[2] = -b * y[2] - y[1] + a * y[0] - a * (y[0] ** 3)


def integrator_rk4(y_curr, params, dt, n, stride, kneadings_start, kneadings_end):
    first_derivative_prev = np.zeros(DIM)
    first_derivative_curr = np.zeros(DIM)
    kneading_index = 0
    kneadings_weighted_sum = 0

    rhs(params, y_curr, first_derivative_prev)

    for i in range(1, n):
        for j in range(stride):
            stepper_rk4(params, y_curr, dt)

        for k in range(DIM):
            if y_curr[k] > INFINITY or y_curr[k] < -INFINITY:
                return InfinityError

        rhs(params, y_curr, first_derivative_curr)
        if first_derivative_prev[0] * first_derivative_curr[0] < 0:
            if first_derivative_curr[1] < 0 and y_curr[0] > 1:
                if kneading_index >= kneadings_start:
                    kneadings_weighted_sum += 1 / (2.0 ** (-kneading_index + kneadings_end + 1))
                kneading_index += 1
            elif first_derivative_curr[1] > 0 and y_curr[0] < -1:
                kneading_index += 1

        first_derivative_prev[:] = first_derivative_curr

    if kneading_index > kneadings_end:
        return kneadings_weighted_sum

    return KneadingDoNotEndError


def sweep(kneadings_weighted_sum_set,
          a_start,
          a_end,
          a_count,
          b_start,
          b_end,
          b_count,
          dt,
          n,
          stride,
          kneadings_start,
          kneadings_end):
    a_step = (a_end - a_start) / (a_count - 1)
    b_step = (b_end - b_start) / (b_count - 1)

    results = []

    for idx in range(a_count * b_count):
        i = idx // a_count
        j = idx % a_count

        params = np.array([a_start + i * a_step, b_start + j * b_step], dtype=np.float32)
        y_init = np.array([1e-8, 0.0, 0.0], dtype=np.float32)

        result = integrator_rk4(y_init.copy(), params, dt, n, stride, kneadings_start, kneadings_end)

        # Сохраняем результаты вместе с параметрами
        results.append((params[0], params[1], result))

    return results


if __name__ == "__main__":
    dt = 0.01
    n = 30000
    stride = 1
    max_kneadings = 20
    sweep_size = 5
    kneadings_weighted_sum_set = np.zeros(sweep_size * sweep_size)

    a_start = 0.0
    a_end = 2.2
    b_start = 0
    b_end = 1.5

    results = sweep(
        kneadings_weighted_sum_set,
        a_start,
        a_end,
        sweep_size,
        b_start,
        b_end,
        sweep_size,
        dt,
        n,
        stride,
        0,
        10,
    )

    # Вывод результатов в двоичном формате
    print("Results:")

    for a_val, b_val, result in results:
        # Преобразуем результат в двоичный формат и выводим с параметрами
        binary_result = format(int(result), 'b') if result not in [InfinityError, KneadingDoNotEndError] else 'Error'

        print(f"a: {a_val:.2f}, b: {b_val:.2f} => Result: {binary_result} (Raw: {result})")