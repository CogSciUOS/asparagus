import doctest
import time


def func(arg1, arg2):
    """Summary line.

    Extended description of function.

    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2

    Returns:
        bool: Description of return value

    """
    return True


def get_current_time():
    """Gets the current time (seconds that have past since 1970).

    Return the time in seconds since the epoch as a floating point number.
    Note that even though the time is always returned as a floating point number,
    not all systems provide time with a better precision than 1 second.

    Returns:
        float: seconds since the epoch


    >>> get_current_time() > 0
    True
    """

    return time.time()


if __name__ == "__main__":
    doctest.testmod()
    print("Current seconds since epoch", get_current_time())
