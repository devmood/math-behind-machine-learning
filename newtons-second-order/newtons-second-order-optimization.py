from sympy import *
from sympy.parsing import sympy_parser as spp
import numpy as np
import matplotlib.pyplot as plt


matrix = Matrix(symbols('x1 x2'))
plot_from, plot_to, plot_step = -7.0, 7.0, 0.1
target_precision = 0.3


def dfdx(x, g):
    return [float(g[i].subs(matrix[0], x[0]).subs(matrix[1], x[1])) for i in range(len(g))]


def newtons_optimization_method(x, x_result, function):
    jacobian = [diff(function, i) for i in matrix]
    print(jacobian)
    print(matrix)
    hessian = Matrix([[diff(jacobian[j], matrix[i]) for i in range(len(matrix))]
                      for j in range(len(jacobian))])
    hessian_inv = hessian.inv()
    print(hessian, '\n')
    print(hessian_inv, '\n')

    xn = [[0, 0]]
    xn[0] = x

    i = 0
    print(xn)
    try:
        while np.linalg.norm(xn[-1] - x_result) > target_precision:
            gn = Matrix(dfdx(xn[i], jacobian))
            delta_xn = -hessian_inv * gn
            delta_xn = delta_xn.subs(matrix[0], xn[i][0]).subs(matrix[1], xn[i][1])
            xn.append(Matrix(xn[i]) + delta_xn)
            i += 1
        # print('newtons method, result distance:', np.linalg.norm(xn[-1] - x_result)

    except:
        pass

    xn = np.array(xn)
    plt.plot(xn[:, 0], xn[:, 1], 'k-o')


if __name__ == ("__main__"):
    x = [-4.0, 6.0] # start location whatever it is lol

    function = spp.parse_expr('x1**2 - x2 * x1 - x1 + 4 * x2**2')
    x_result = np.array([[0, 0]])
    print('thisiiisisiis', x_result)

    i1 = np.arange(plot_from, plot_to, plot_step)
    i2 = np.arange(plot_from, plot_to, plot_step)
    x1_mesh, x2_mesh = np.meshgrid(i1, i2)
    f_str = function.__str__().replace('x1', 'x1_mesh').replace('x2', 'x2_mesh')
    f_mesh = eval(f_str)

    plt.figure()
    plt.imshow(f_mesh, cmap='Paired', origin='lower',
        extent=[plot_from - 20, plot_to + 20, plot_from - 20, plot_to + 20])
    plt.colorbar()
    plt.title('f(x) =' + str(function))
    plt.xlabel('x1')
    plt.ylabel('x2')

    newtons_optimization_method(x, x_result, str(function))
    plt.show()
