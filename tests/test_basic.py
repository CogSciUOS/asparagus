import pytest
import good_style


def test_good_style_time_greater_zero():
    assert(good_style.get_current_time() > 0)
