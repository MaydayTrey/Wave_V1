import numpy as np
import time
from functools import wraps
from config_wave import config
Diagnostic_Dictionary = {}

def Diagnostic_Decorator(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = function(*args, **kwargs)
        elapsed = time.perf_counter() - start

        if isinstance(result, np.ndarray):
            Diagnostic_Dictionary.setdefault(function.__name__, []).append({
                "Shape": result.shape,
                "NaN": np.isnan(result).any(),
                "Type": type(result).__name__,
                "Latency": elapsed
            })
        else:
            Diagnostic_Dictionary.setdefault(function.__name__, []).append({
                "Shape": None,
                "NaN": None,
                "Type": type(result).__name__,
                "Latency": elapsed
            })
            if config.STRICT_MODE:
                raise TypeError(f"{function.__name__} returned non-ndarray: {type(result)}")

        return result
    return wrapper

