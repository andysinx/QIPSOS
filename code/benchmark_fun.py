import numpy as np

# =========================
# BENCHMARK FUNCTIONS
# =========================

def beale(x):
    x1, x2 = x
    return (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2


def easom(x):
    x1, x2 = x
    return -np.cos(x1)*np.cos(x2)*np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))


def matyas(x):
    x1, x2 = x
    return 0.26*(x1**2 + x2**2) - 0.48*x1*x2


def bohachevsky1(x):
    x1, x2 = x
    return x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1) - 0.4*np.cos(4*np.pi*x2) + 0.7


def bohachevsky2(x):
    x1, x2 = x
    return x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1)*np.cos(4*np.pi*x2) + 0.3


def bohachevsky3(x):
    x1, x2 = x
    return x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1 + 4*np.pi*x2) + 0.3


def booth(x):
    x1, x2 = x
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2


def schaffer(x):
    x1, x2 = x
    num = np.sin(np.sqrt(x1**2 + x2**2))**2 - 0.5
    den = (1 + 0.001*(x1**2 + x2**2))**2
    return 0.5 + num/den


def six_hump_camel(x):
    x1, x2 = x
    return (4 - 2.1*x1**2 + (x1**4)/3)*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2


def shubert(x):
    x1, x2 = x
    s1 = sum(i*np.cos((i+1)*x1 + i) for i in range(1, 6))
    s2 = sum(i*np.cos((i+1)*x2 + i) for i in range(1, 6))
    return s1 * s2


def colville(x):
    x1, x2, x3, x4 = x
    return (
        100*(x1**2 - x2)**2 +
        (x1 - 1)**2 +
        (x3 - 1)**2 +
        90*(x3**2 - x4)**2 +
        10.1*(x2 - 1)**2 +
        (x4 - 1)**2 +
        19.8*(x2 - 1)*(x4 - 1)
    )

def sphere(x):
    return np.sum(x**2)


def rastrigin(x):
    return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10)


def ackley(x):
    d = len(x)
    return -20*np.exp(-0.2*np.sqrt(np.sum(x**2)/d)) \
           - np.exp(np.sum(np.cos(2*np.pi*x))/d) + 20 + np.e


def rosenbrock(x):
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)


def griewank(x):
    return np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1)))) + 1


def sumsquares(x):
    return np.sum((np.arange(1, len(x)+1)) * x**2)


def quartic(x):
    return np.sum((np.arange(1, len(x)+1)) * x**4)


def step(x):
    return np.sum((x + 0.5)**2)


def zakharov(x):
    d = len(x)
    s1 = np.sum(x**2)
    s2 = np.sum(0.5*np.arange(1, d+1)*x)
    return s1 + s2**2 + s2**4


def schwefel_222(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def schwefel_12(x):
    return np.sum([np.sum(x[:i])**2 for i in range(1, len(x)+1)])


def dixon_price(x):
    return (x[0]-1)**2 + np.sum((np.arange(2, len(x)+1))*(2*x[1:]**2 - x[:-1])**2)


def michalewicz(x):
    m = 20
    return -np.sum(np.sin(x) * (np.sin((np.arange(1, len(x)+1)*x**2)/np.pi))**(2*m))


def benchmark_problems():
    return {
            # 2D
            "Beale": (beale, 2, [(-4.5, 4.5)]),
            "Easom": (easom, 2, [(-100,100)]),
            "Matyas": (matyas, 2, [(-10, 10)]),
            "Booth": (booth, 2, [(-10, 10)]),
            "Schaffer": (schaffer, 2, [(-100, 100)]),
            "SixHump": (six_hump_camel, 2, [(-5, 5)]),
            "Shubert": (shubert, 2, [(-10, 10)]),
            "Colville": (colville, 4, [(-10, 10)]),

            # Bohachevsky
            "B1": (bohachevsky1, 2, [(-100, 100)]),
            "B2": (bohachevsky2, 2, [(-100, 100)]),
            "B3": (bohachevsky3, 2, [(-100, 100)]),

            # D-DIM
            "Sphere": (sphere, 30, [(-100, 100)]),
            "Rastrigin": (rastrigin, 30, [(-5.12, 5.12)]),
            "Ackley": (ackley, 30, [(-32, 32)]),
            "Rosenbrock": (rosenbrock, 2, [(-30, 30)]),
            "Griewank": (griewank, 30, [(-600, 600)]),

            "SumSquares": (sumsquares, 30, [(-10, 10)]),
            "Quartic": (quartic, 30, [(-1.28, 1.28)]),
            "Step": (step, 30, [(-5.12, 5.12)]),

            "Zakharov": (zakharov, 10, [(-5, 10)]),

            "Schwefel222": (schwefel_222, 30, [(-10, 10)]),
            "Schwefel12": (schwefel_12, 30, [(-100, 100)]),

            "DixonPrice": (dixon_price, 30, [(-10, 10)]),

            "Michalewicz2": (michalewicz, 2, [(0, np.pi)]),
            "Michalewicz5": (michalewicz, 5, [(0, np.pi)]),
            "Michalewicz10": (michalewicz, 10, [(0, np.pi)]),
        }

def benchmark_problems1():
    return {
        # 2D
        "Beale": (beale, 2, [(-4.5, 4.5), (-4.5, 4.5)]),
        "Easom": (easom, 2, [(-100, 100), (-100, 100)]),
        "Matyas": (matyas, 2, [(-10, 10), (-10, 10)]),
        "Booth": (booth, 2, [(-10, 10), (-10, 10)]),
        "Schaffer": (schaffer, 2, [(-100, 100), (-100, 100)]),
        "SixHump": (six_hump_camel, 2, [(-5, 5), (-5, 5)]),
        "Shubert": (shubert, 2, [(-10, 10), (-10, 10)]),

        # 4D
        "Colville": (colville, 4, [(-10, 10)] * 4),

        # Bohachevsky
        "B1": (bohachevsky1, 2, [(-100, 100), (-100, 100)]),
        "B2": (bohachevsky2, 2, [(-100, 100), (-100, 100)]),
        "B3": (bohachevsky3, 2, [(-100, 100), (-100, 100)]),

        # D-DIM
        "Sphere": (sphere, 30, [(-100, 100)] * 30),
        "Rastrigin": (rastrigin, 30, [(-5.12, 5.12)] * 30),
        "Ackley": (ackley, 30, [(-32, 32)] * 30),
        "Rosenbrock": (rosenbrock, 30, [(-30, 30)] * 30),
        "Griewank": (griewank, 30, [(-600, 600)] * 30),

        "SumSquares": (sumsquares, 30, [(-10, 10)] * 30),
        "Quartic": (quartic, 30, [(-1.28, 1.28)] * 30),
        "Step": (step, 30, [(-5.12, 5.12)] * 30),

        "Zakharov": (zakharov, 10, [(-5, 10)] * 10),

        "Schwefel222": (schwefel_222, 30, [(-10, 10)] * 30),
        "Schwefel12": (schwefel_12, 30, [(-100, 100)] * 30),

        "DixonPrice": (dixon_price, 30, [(-10, 10)] * 30),

        "Michalewicz2": (michalewicz, 2, [(0, np.pi)] * 2),
        "Michalewicz5": (michalewicz, 5, [(0, np.pi)] * 5),
        "Michalewicz10": (michalewicz, 10, [(0, np.pi)] * 10),
    }