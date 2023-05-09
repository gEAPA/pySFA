import numpy as np
from ..constant import FUN_COST


def assert_valid_basic_data(y, x, fun):
    """Throw a ValueError if input data are invalid.
    Args:
        y : {ndarray}
        x : {ndarray} The input data.
    Returns:
        lists of y and x. If 'fun' is cost function, returns -y and -x
    """
    if fun == FUN_COST:
        y = -1 * y
        x = -1 * x

    y = trans_list(y)
    x = trans_list(x)

    y = to_1d_list(y)
    x = to_2d_list(x)

    y_shape = np.asarray(y).shape
    x_shape = np.asarray(x).shape

    if len(y_shape) == 2 and y_shape[1] != 1:
        raise ValueError(
            "The output must be one dimensional array.")

    if y_shape[0] != x_shape[0]:
        raise ValueError(
            "x and y must have the same length.")

    return y, x


def trans_list(li):
    if type(li) == list:
        return li
    return li.tolist()


def to_1d_list(li):
    if type(li) == int or type(li) == float:
        return [li]
    if type(li[0]) == list:
        rl = []
        for i in range(len(li)):
            rl.append(li[i][0])
        return rl
    return li


def to_2d_list(li):
    if type(li[0]) != list:
        rl = []
        for value in li:
            rl.append([value])
        return rl
    return li
