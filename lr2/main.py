import logging
import numpy as np

from scipy.integrate import quad, trapezoid, simpson
from typing import Callable
from matplotlib import pyplot as plt


class DataIntegrals:

    def __init__(self, maxiter: int, res: float):
        self.maxiter = maxiter
        self.res = res


def f(x: float) -> float:
    return np.sin(x) - 1/x


def self_made_quad(f: Callable[[float], float], bracket: tuple, maxiter: int = 50) -> float:
    a, b = bracket
    iter = (b - a) / maxiter 
    s = 0

    while a < b:
        a += iter
        if a > b:
            break
        s += iter * f(a)

    return s


def self_made_trapezoid(f: Callable[[float], float], bracket: tuple, maxiter: int = 50) -> float:
    a, b = bracket
    iter = (b - a) / maxiter
    s = 0

    while a < b:
        s += 1/2 * (f(a) + f(a + iter)) * iter
        a += iter 
    
    return s


def self_made_simpson(f: Callable[[float], float], bracket: tuple, maxiter: int = 50) -> float:
    a, b = bracket
    h = (b - a) / (2 * maxiter)
    s = f(a) + f(b)
    iter = 1

    while a < b:
        a += h
        if a >= b:
            break

        if iter % 2 == 0:
            s += 2 * f(a)

        else:
            s += 4 * f(a)

        iter += 1

    return s * h / 3


def base_func_graph(f: Callable[[float], float]) -> None:
    x_left = np.linspace(-5, -0.01, 50)
    x_right = np.linspace(0.01, 5, 50)

    plt.plot(x_left, f(x_left), color='blue')
    plt.plot(x_right, f(x_right), color='blue')
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.savefig("Base_Graphic.png")
    plt.close()


def creating_graphs_of_iterations(iterations: list[DataIntegrals], title: str = "test") -> None:
    x = []
    iters = []

    for data in iterations:
        x.append(data.res)
        iters.append(data.maxiter)

    plt.plot(iters, x)
    plt.grid()
    plt.xlabel("iter")
    plt.ylabel("x")
    plt.savefig(f"{title.replace(" ", "_")}.png")
    plt.close()


def creating_graphs_of_errors(iterations: list[DataIntegrals], analytic_value: float, title: str = "test") -> None:
    errors = []
    iters = []

    for iter in iterations:
        errors.append(abs(iter.res - analytic_value))
        iters.append(iter.maxiter)

    plt.plot(iters, errors)
    plt.grid()
    plt.xlabel("iter")
    plt.ylabel("errors")
    plt.savefig(f"{title.replace(" ", "_")}.png")
    plt.close()   


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=r"[%(levelname)s] %(message)s"
    )

    base_func_graph(f)

    bracket = (0.1, 1.6)
    iters = [10, 50, 100, 150, 200]
    quad_data: list[DataIntegrals] = []
    trapezoid_data: list[DataIntegrals] = []
    simpson_data: list[DataIntegrals] = []

    logging.info("[ЭТАП 1] РАССМАТРИВАЮТСЯ МЕТОДЫ ЧИСЛЕННОГО ИНТЕГРИРОВАНИЯ, ВЫПОЛНЕННЫЕ БРИГАДОЙ")

    for iter in iters:
        data = DataIntegrals(maxiter=iter, res=self_made_quad(f, bracket, iter))
        quad_data.append(data)

        data = DataIntegrals(maxiter=iter, res=self_made_trapezoid(f, bracket, iter))
        trapezoid_data.append(data)


        data = DataIntegrals(maxiter=iter, res=self_made_simpson(f, bracket, iter))
        simpson_data.append(data)


    logging.info(f"""Метод прямоугольников(рассматривали метод правых прямоугольников):
                    Кол-во делений: {quad_data[-1].maxiter};
                    Интеграл: {quad_data[-1].res}.\n""")
    logging.info(f"""Метод трапеции:
                    Кол-во делений: {trapezoid_data[-1].maxiter};
                    Интеграл: {trapezoid_data[-1].res}. \n""")
    logging.info(f"""Метод Симпсона:
                    Кол-во делений: {simpson_data[-1].maxiter};
                    Интеграл: {simpson_data[-1].res}""")


    logging.info("[ЭТАП 2] РАССМАТРИВАЮТСЯ ТЕ ЖЕ МЕТОДЫ ЧИСЛЕННОГО ИНТЕГРИРОВАНИЯ, НО ВЗЯТЫЕ ИЗ БИБЛИОТЕКИ SCIPY")

    logging.info("Вывод из метода прямоугольников:")
    print(quad(f, 0.1, 1.6))

    x = np.linspace(0.1, 1.6, 50)

    logging.info("Вывод из метода трапеций:")
    print(trapezoid(f(x), x))

    logging.info("Вывод из метода Симпсона")
    print(simpson(f(x), x))

    creating_graphs_of_iterations(quad_data, "Quad Depends From Iters")
    creating_graphs_of_iterations(trapezoid_data, "Trapezoid Depends From Iters")
    creating_graphs_of_iterations(simpson_data, "Simpson Depends From Iters")

    creating_graphs_of_errors(quad_data, -1.748, "Quad Errors")
    creating_graphs_of_errors(trapezoid_data, -1.748, "Trapezoid Errors")
    creating_graphs_of_errors(simpson_data, -1.748, "Simpson Errors")