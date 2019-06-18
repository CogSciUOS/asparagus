import pytest
import math

#good_style = __import__("code.good_style.py")

from code.good_style import get_current_time


def test_good_style_time_greater_zero():
    assert(get_current_time() > 0)


def test_square_of_2():
    assert(math.sqrt(4) == 2)
