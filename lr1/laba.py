import numpy as np
import matplotlib.pyplot as plt
import logging

from scipy.optimize import root_scalar, RootResults
from typing import Callable, Any

class MethodClass:

    def __init__(self, method: str,  epsilon: float, root: float, success: bool, iterations: list):
        self.method = method
        self.root = root
        self.success = success
        self.iterations = iterations
        self.epsilon = epsilon


    def __str__(self):
        return f"""
                method: {self.method}
                epsilon: {self.epsilon}
                root: {self.root}
                success: {self.success}
                iterations: {self.iterations}
                """


def f(x: float) -> float:
    return 0.5 * x ** 3 - x - 2

def df_analyze(x: float) -> float:
    return 1.5 * x ** 2 - 1

def df_digit(f: Callable[[float], float], x: float) -> float:
    delta_x = 10**-6
    return (f(delta_x/2 + x) - f(x - delta_x / 2)) / delta_x

def self_made_bisect(bracket: tuple, epsilon: float, maxiter: int) -> dict:
    a, b = bracket

    if f(a) * f(b) > 0: #Данное условие говорит о том, что если оно больше нуля, то корня тут быть не может и нужно менять интервал
        return {
            "method": "bisection",
            "root": None,
            "success": False,
            "iterations": []
        }

    iterations = []

    for i in range(maxiter):
        x = (a + b) / 2

        iterations.append({ 
            "iter": i + 1,
            "a": a,
            "b": b,
            "x": x,
            "f(x)": f(x)
        }) # Добавляем каждую итерацию для дальнейших графиков

        if f(x) == 0: #Если значение функции равно нулю, то заканчиваем итерацию(идеальное попадание)
            return {
                "method": "bisection",
                "root": x,
                "success": True,
                "iterations": iterations
            }

        if abs(b - a) < epsilon or abs(f(b) - f(a)) < epsilon: #Если выполняется критерий приближения к корню внутри интервала, то заканчиваем итерацию
            return {
                "method": "bisection",
                "root": x,
                "success": True,
                "iterations": iterations
            }

        if f(x) * f(a) < 0: #Шаг бисекции
            b = x
        else:
            a = x

    return {
        "method": "bisection",
        "root": (a + b) / 2,
        "success": False,
        "iterations": iterations
    }
        

def self_made_newton(bracket: tuple, epsilon: float, maxiter: int) -> dict:
    x = bracket[-1]  # начальная точка

    iterations = []

    for i in range(maxiter):
        fx = f(x)
        dfx = df_digit(f, x)

        # защита от деления на 0
        if dfx == 0:
            return {
                "method": "newton",
                "root": None,
                "success": False,
                "iterations": iterations
            }

        x_new = x - fx / dfx

        # сохраняем итерацию
        iterations.append({
            "iter": i + 1,
            "x": x_new,
            "f(x)": f(x_new)
        })

        # критерии остановки
        if abs(x_new - x) < epsilon or abs(f(x_new)) < epsilon:
            return {
                "method": "newton",
                "root": x_new,
                "success": True,
                "iterations": iterations
            }

        x = x_new

    # если не сошлось
    return {
        "method": "newton",
        "root": x,
        "success": False,
        "iterations": iterations
    }


def bisect(f: Callable[[float], float], bracket: tuple, epsilon: float, M: int) -> RootResults:
    return root_scalar(f, method="bisect", bracket=bracket, xtol=epsilon, maxiter=M)


def newton(f: Callable[[float], float], bracket: tuple, epsilon: float, M: int) -> RootResults:
    return root_scalar(f, method="newton", bracket=bracket, xtol=epsilon, maxiter=M, x0=bracket[0])


def function_graphic(f: Callable[[float], float]) -> None:
    x = np.linspace(-6, 4, 100)
    y = f(x)

    plt.plot(x, y)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("function_graphic.png")
    plt.close()
    # plt.show()


def convergence_of_the_algorithm(iterations: list, title: str) -> None: #Графики сходимости алгоритма
    iter = []
    x = []
    for elem in iterations: #Собираем итератор в оси X, а сами значения x в оси Y(как бы странно это не звучало)
        iter.append(elem["iter"])
        x.append(f(elem["x"]))

    plt.title(title)
    plt.plot(iter, x)
    plt.grid()
    plt.xlabel("iter")
    plt.ylabel("f(x)")
    plt.savefig(f"{title.replace(" ", "_")}.png")
    plt.close()


def error_of_algorithm(iterations: list, title: str) -> None:
    errors = []

    for i in range(1, len(iterations)):
        errors.append(abs(iterations[i]["x"] - iterations[i-1]["x"]))

    plt.title(title)
    plt.plot(range(1, len(errors)+1), errors)
    plt.grid()
    plt.xlabel("iter")
    plt.ylabel("error")
    plt.savefig(f"{title.replace(" ", "_")}.png")
    plt.close()

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=r"[%(levelname)s] %(message)s"
    )

    bracket = (-6, 4)
    epsilon = 0.00000001
    M = 100
    function_graphic(f=f)

    with open("logfile.txt", "w", encoding="utf-8") as file:
        logging.info("[ЭТАП 1] РАССМАТРИВАЮТСЯ МЕТОДЫ, НАПИСАННЫЕ БРИГАДОЙ")
        logging.info("Рассматривается метод половинного деления.")
        bisect_alg = self_made_bisect(bracket=bracket, epsilon=epsilon, maxiter=M)
        bisect_obj = MethodClass(method=bisect_alg["method"],
                                 epsilon=epsilon,
                                 root=bisect_alg["root"],
                                 success=bisect_alg["success"],
                                 iterations=bisect_alg["iterations"])
        print(str(bisect_obj) + "\n")
        file.write(f"""bisect method:
                        results: {str(bisect_obj)}
                        \n""")

        try:
            convergence_of_the_algorithm(bisect_alg["iterations"], "Bisect Algorithm")
            error_of_algorithm(bisect_alg["iterations"], "Errors of Bisect Algorithm")
        except KeyError as e:
            logging.error(e)

        logging.info("Рассматривается метод Ньютона.")
        newton_alg = self_made_newton(bracket=bracket, epsilon=epsilon, maxiter=M)
        newton_obj = MethodClass(method=newton_alg["method"],
                                 epsilon=epsilon,
                                 root=newton_alg["root"],
                                 success=newton_alg["success"],
                                 iterations=newton_alg["iterations"])
        print(str(newton_obj) + "\n")
        file.write(f"""newtone method:
                        results: {str(newton_obj)}
                        \n""")

        try:
            convergence_of_the_algorithm(newton_alg["iterations"], "Newton Algorithm")
            error_of_algorithm(newton_alg["iterations"], "Errors of Newton Algorithm")
        except KeyError as e:
            logging.error(e) 

        bisect_scipy = bisect(f=f, bracket=bracket, epsilon=epsilon, M=M)
        logging.info("[ЭТАП 2] РАССМАТРИВАЮТСЯ ВЫШЕУКАЗАННЫЕ МЕТОДЫ, ВЗЯТЫЕ ИЗ БИБЛИОТЕКИ SCIPY")
        logging.info("Рассматривается метод половинного деления.")
        print(str(bisect_scipy) + "\n")
        file.write(f"""Bisect Scipy Method:
                        epsilon: {epsilon};
                        results: {str(bisect_scipy)}""")

        newton_scipy = newton(f=f, bracket=bracket, epsilon=epsilon, M=M)
        logging.info("Рассматривается метод Ньютона.")
        print(str(newton_scipy))
        file.write(f"""Newton Scipy Method:
                        epsilon: {epsilon};
                        results: {str(newton_scipy)}""")



    

