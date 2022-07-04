import time


def time_counter(func):
    def fun(*args, **argv):
        t = time.perf_counter()
        result = func(*args, **argv)
        print(f"function '{func.__name__}' cost {time.perf_counter()-t}. ")
        return result

    return fun
