import numpy as np

def monte_carlo_execute(func, bounds, dtype, n=100):
    # print(bounds)
    rnd = [np.random.uniform(b_l, b_h, n).tolist() for b_l, b_h in bounds]
    rnd_choices = [
        [rnd[i][np.random.randint(0, n)] for i in range(len(bounds))]
        for _ in range(n)
    ]

    return np.array(rnd_choices), np.array([func(r) for r in rnd_choices], dtype=dtype)

def monte_carlo_bounds(func, bounds, dtype, n=100, maxiter=100, tops=10, ):
    iter = 0
    while iter < maxiter:
        func_in, func_out  = monte_carlo_execute(func, bounds, dtype, n)
        idx = np.argpartition(func_out, -tops, order=[d[0] for d in dtype])[-tops:]
        bounds_sample = func_in[idx]
        # print("bounds_sample", bounds_sample)
        # print("func_out", func_out[idx])
        # print("idx", idx)
        new_bounds = list(zip(np.min(bounds_sample, axis=0), np.max(bounds_sample, axis=0)))
        # print(new_bounds, func_in)
        assert len(new_bounds) == len(new_bounds)
        bounds = new_bounds
        iter += 1

    return bounds
