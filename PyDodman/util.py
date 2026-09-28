import math
import sys
import os
from dotenv import load_dotenv
# loading variables from .env file
load_dotenv() 

def _env_or_use_default(key: str, default: float) -> float:
    print(key)
    print(os.getenv(key))
    try:
        key_val: float = float(os.getenv(key))
        return key_val
    except:
        return default

RELATIVE_TOLERANCE: float = _env_or_use_default("DODMAN_RELATIVE_TOLERANCE",1e-12)
EPSILON: float = _env_or_use_default("DODMAN_EPSILON",sys.float_info.epsilon)





def is_almost_equal(a: float|int, b:float|int, rel_tol: float=RELATIVE_TOLERANCE, abs_tol: float=EPSILON) -> bool:

    """Uses the math package's isclose function to return True if values a and be are close and False otherwise.

    Args:
        a (float | int): Value to compare to b.
        b (float | int): Value to compare to a.
        rel_tol (float, optional): relative tolerance – it is the maximum allowed difference between a and b, relative to the larger absolute value of a or b. Defaults to RELATIVE_TOLERANCE.
        abs_tol (float, optional): The minimum absolute tolerance – useful for comparisons near zero. abs_tol must be at least zero. Defaults to EPSILON.

    Returns:
        bool: True if a and b are close and false otherwise.
    """
    return math.isclose(a,b,rel_tol=rel_tol,abs_tol=abs_tol)