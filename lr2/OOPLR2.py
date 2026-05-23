import numpy as np

from abc import ABC, abstractmethod
from scipy.integrate import quad, trapezoid, simpson
from typing import Callable
from matplotlib import pyplot as plt


def func(x: float | np.ndarray) -> float | np.ndarray:
    return np.sin(x) - 1/x


class Statistic:
    def __init__(self, n: int, delta: float, result: float):
        self.n = n
        self.delta = delta
        self.result = result


class FunctionPlotter:
    def __init__(self, func: Callable[[float], float]):
        self.func = func

    def create_function_plot(self, a: float, b: float) -> None:
        x = np.linspace(a, b, 500)
        plt.plot(x, self.func(x), color="blue")
        plt.grid()
        plt.title("Base Graphic")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.savefig("Base_Graphic.png")
        plt.close()


    def integration_plot(self, statistic: list[Statistic], title: str = "Integral") -> None:
        result = [stat.result for stat in statistic]
        n = [stat.n for stat in statistic]

        plt.plot(n, result, color="blue")
        plt.grid()
        plt.title("Интеграл по количеству дискретизаций")
        plt.xscale("log")
        # plt.yscale("log")
        plt.xlabel("Дискретизация")
        plt.ylabel("Интеграл")
        plt.savefig(f"{title.replace(" ", "_")}.png")
        plt.close()


    def error_plot(self, statistic: list[Statistic], title: str = "Error") -> None:
        delta = [stat.delta for stat in statistic]
        n = [stat.n for stat in statistic]

        plt.plot(n, delta, color="blue")
        plt.grid()
        plt.title("Погрешность по кол-ву дискретизации")
        plt.xscale("log")
        # plt.yscale("log")
        plt.xlabel("Дискретизация")
        plt.ylabel(r"Погрешность $\delta$")
        plt.savefig(f"{title.replace(" ", "_")}.png")
        plt.close()

class BaseIntegrator(ABC):
    def __init__(self):
        self._statisticlist: list[Statistic] = []

    @property
    def statistic(self) -> list[Statistic]:
        return self._statisticlist

    @abstractmethod
    def integrate(self, func: Callable[[float], float], a: float, b: float, epsilon: float) -> float:
        ...


class RectIntegrator(BaseIntegrator):
    def integrate(self, func: Callable[[float], float], a: float, b: float, epsilon: float) -> float:
        n = 2
        delta: float | None = None
        result_prev: float | None = None

        while delta is None or delta > epsilon:
            h = (b - a) / n
            x = a + h / 2
            result = 0
            
            while x < b:
                result += func(x) * h
                x += h
            
            if result_prev is not None:
                delta = abs(result - result_prev)
            
            result_prev = result
            n *= 2 
            self._statisticlist.append(Statistic(n, delta, result))
        
        assert result_prev is not None
        return result_prev


class TrapezoidIntegrator(BaseIntegrator):
    def integrate(self, func: Callable[[float], float], a: float, b: float, epsilon: float) -> float:
        n = 2
        delta: float | None = None
        result_prev: float | None = None

        while delta is None or delta > epsilon:
            h = (b - a) / n
            x = a
            result = (func(a) + func(b)) / 2 
            
            for i in range(1, n):
                x = a + i * h
                result += func(x)
            
            result *= h
            
            if result_prev is not None:
                delta = abs(result - result_prev)
            
            result_prev = result
            n *= 2 
            self._statisticlist.append(Statistic(n, delta, result))
        
        assert result_prev is not None
        return result_prev

class SimpsonIntegrator(BaseIntegrator):
    def integrate(self, func: Callable[[float], float], a: float, b: float, epsilon: float) -> float:
        n = 2
        delta: float | None = None
        result_prev: float | None = None

        while delta is None or delta > epsilon:
            h = (b - a) / n
            result = func(a) + func(b)
            
            x = a + h
            while x < b:
                result += 4 * func(x)
                x += 2 * h
            
            x = a + 2 * h
            while x < b - h/2:
                result += 2 * func(x)
                x += 2 * h
            
            result *= h / 3
            
            if result_prev is not None:
                delta = abs(result - result_prev)
            
            result_prev = result
            n *= 2
            self._statisticlist.append(Statistic(n, delta, result))
        
        assert result_prev is not None
        return result_prev
    

if __name__ == "__main__":
    a, b = 0.1, 1.6
    epsilon = 1e-5

    plotter = FunctionPlotter(func)
    plotter.create_function_plot(a, b)

    print("Self made integrals:")
    print("Rect Method")
    rect = RectIntegrator()
    print(rect.integrate(func, a, b, epsilon))

    print("Trapezoid Method")
    trapez = TrapezoidIntegrator()
    print(trapez.integrate(func, a, b, epsilon))

    print("Simpson Method")
    simpson_integrator = SimpsonIntegrator()
    print(simpson_integrator.integrate(func, a, b, epsilon))

    plotter.integration_plot(rect.statistic, "RectIntegral")
    plotter.integration_plot(trapez.statistic, "TrapezIntegral")
    plotter.integration_plot(simpson_integrator.statistic, "SimpsonIntegral")

    plotter.error_plot(rect.statistic, "RectError")
    plotter.error_plot(trapez.statistic, "TrapezError")
    plotter.error_plot(simpson_integrator.statistic, "SimpsonError")

    print("Scipy methods:")
    x = np.linspace(0.1, 1.6, 50)

    print("Rect Method")
    print(quad(func, 0.1, 1.6)[0])

    print("Trapezoid Method")
    print(trapezoid(func(x), x))

    print("Simpson Method")
    print(simpson(func(x), x))


