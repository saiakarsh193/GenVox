import time
from functools import wraps
from typing import Callable

def function_timer(func: Callable) -> Callable:
    """
    function_timer is a decorator that measures the about of time a function/method takes to run
    """
    @wraps(func)
    def inner_function(*args, **kwargs):
        st = time.time()
        return_value = func(*args, **kwargs)
        en = time.time()
        print(f"{func.__name__}() time taken: {en - st}")
        return return_value
    return inner_function
