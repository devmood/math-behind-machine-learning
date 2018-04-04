def function(y, x):
    """
    Playing with ASCII code to count y for a given x.
    The function is given as a string.
    It will not work for neither fractals nor any other weird stuff
    such as square roots, etc.
    Acceptable: 12x^4-17x+4 and similar
    please write 1x instead of x by itself as for now
    """
    numbers_list = []
    temp = ""
    index = -1

    while len(y):
        # if there are x's in the function string
        if y.find('x') > 0:
            temp = y[:y.find('x')]
            y = y[y.find('x'):]
            numbers_list.append(float(temp))
            index += 1
            if y[:2] == "x^":
                temp = ""
                i = 1
                while i < len(y):
                    i += 1
                    if y[i].isdigit() or (i == 2 and y[i] == '-'):
                        temp += y[i]
                    elif i == len(y) - 1:
                        y = ""
                        numbers_list[index] *= (x ** float(temp))
                        break
                    elif not y[i].isdigit():
                        y = y[i:]
                        numbers_list[index] *= (x ** float(temp))
                        break
                temp = ""
            else:
                numbers_list[index] *= x
                y = y[1:]
        else:
            temp = y
            numbers_list.append(float(temp))
            break

    result = 0
    for numb in numbers_list:
        result += numb
    return result


def f(y, x):
    return function(y, x)


def df(x):
    return 6 * x - 7


# delta shit i suppose
def dx(y, x):
    return abs(0 - f(y, x))


def find_roots(y, x, threshold):
    delta = dx(y, x)
    while delta > threshold:
        x = x - f(y, x) / df(x)
        delta = dx(y, x)
    print('root:', x, 'value at root:', f(y, x))


if __name__ == "__main__":
    for x in [-3, 5]:
        find_roots('3x^2-7x-52', x, 1e-5)
