##### **1. Gradient Descent**
###### as the most common optimization technique in machine learning, which finds the local minima of a function
0. The whole code is being run as follows:
```
if __name__ == '__main__':
    data = some_data # find it out later how to write it in here
    initial_a = 0
    initial_b = 0
    iterations = 1000
    print('Before running the gradient descent algorithm...\na: ', initial_a, ', b: ', initial_b, ',\
    error: ', measure_error(initial_a, initial_b, data))
    a, b = change_coefficients(initial_a, initial_b, iterations)
    print('After running the gradient descent algorithm...\na: ', initial_a, ', b: ', initial_b, ',\
    error: ', measure_error(a, b, data))
```
1. As You may or may not remember the equation of the line is:

![equation of the line][line]

where - a stands for the slope of the line and b is the y-intercept (A.K.A. - bias).
2. In order to find the best fit line for the data, we are looking for the optimal values of a and b, so that the line fits as many points as possible.
3. To measure the error, we can use *Sum of Squared Errors*

![Sum of Squared Errors][sse]

```
def measure_error(a, b, data):
    error = 0
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]
        error += (y - (a*x + b)) ** 2
        # or error += (y - predict()) ** 2
    return error / float(len(data))
```
4. We can run this method repetitively to find the best fit line with the smallest error, we can also plot the graph showing all of the errors for all the coefficients.
```
def change_coefficients(starting_a, starting_b, iterations):
    a = starting_a
    b = starting_b
    for i in range(iterations):
        a, b = step_gradient(a, b, 0.0001, array(data))
    return a, b
```
5. The vlues of the coefficients a and b, for which the graph reaches the minimum are the desired coefficients of the best fit line, You are looking for. It means that distances between all the datapoints and the line will be the smallest.
6. In order to find such coefficients, we need to minimize the function. By counting its partial derivative with respect to a and b seperately.

![derivative of the function with respect to coefficient a - slope][derivative_a]

![derivative of the function with respect to the coefficient b - bias][derivative_b]

```
def step_gradient(a_current, b_current, etha, data):
    a_gradient = 0
    b_gradient = 0
    n = float(len(data))
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]
        a_gradient += -(2/n) * (y - ((b_current * x) + a_current))
        b_gradient += -(2/n) * x * (y - ((b_current * x) + a_current))
    new_a = a_current - (etha * a_gradient)
    new_b = b_current - (etha * b_gradient)
    return new_a, new_b
```

##### **2. Support Vector Machine**
###### supervised machine learning algorithm, which can create a discriminative classifier - optimal hyperplane
0. The w
1. There are two classes, which can be plotted on the 2D graph; moreover - we can 'draw' a line separating both classes from each other - a decision boundry called a *hyperplane*. SVM helps create it.
2. SVMs are used in classification (most often), regression (predicting another outcome in the series of measurments based on the previous ones), outllier prediction (tracking behaviour of some population or whatever else and looking for an anomaly), clustering, etc.
3. SVM needs to learn the relationship between variables, creating the function which is then used to output prediction from some unknown input values.
4. SVMs are already implemented insied sklearn modules, which You can use, however You will firs code it Yourself to learn how it works in the background.
5. SVMs are best for small datasets (up to 1K rows) classification. Other algorithms, such as random forests or deep neural networks require more data; however, almost always come out with a very robust model. Picking the right alogrithm depends on the problem and the data itself.
6. The way how SVMs work is by maximizing the margin between the line and the classes (points which are the closest to the decision boundry - *suport vectors* - they are vectors and they support the creation of the hyperplane). Why maximizing? Because if there's a datapoint which doesn't fall into any of the classes, then it will have the maximum likelyhood of falling into the side of the hyperplane where it should be.
7. Hyperplane is the decision surface. For n dimenensions (i.e. number of features) a hyperplane is always n-1 dimensions. As You can imagine You can't really visualize more than 3 dimensions; however, for computers it's not a problem at all and in machine learning You'll often see that there's quite a few of the dimensions present in most of the problems.
8. Every machine learning model is the function that we want to approximate, its coeffisients are its weights and they're being updated through the optimization process.
9. Preparing the data, plotting it and plotting the naive hyperplane (dummy):

```
import numpy as np
import matplotlib.pyplot as plt

# input data of the form [X, Y, bias]
def prepare_data():
    X = np.array([[-2, 4], [4, 1], [1, 6], [2, 4], [6, 2]])
    # output labels (either -1 or 1)
    y = np.array([-1, -1, 1, 1, 1, 1])
    for d, sample in enumerate(X):
        if d < 2:
            plt.scatter(sample[0], sample[1], marker='-')
        else:
            plt.scatter(sample[0], sample[1], marker='+')
    # plotting a dummy hyperplane, just for sake of visualization
    plt.plot([-2, 6], [6, 0.5])
    return X, y

if __name__ == '__main__':
    X, y = prepare_data()
```

10. The next step is minimizing the loss/error function (c - the loss function)- *hinge loss* - used for maximum-margin classification in SVMs. We always want the result to be positive what denotes the subscript + at the end of the equation (meaning that if something is negative - the result should be 0):

![hinge loss function][hlf]

11. The objective function consists of the loss function (sum of the errors for all of the datapoints) and the first part of the equation, which is a regularizer term.

![objective funtion with regularizer and hinge loss function][objf]

12. In order to optimize the objective function You need to derive the function to get the gradients. Since there are two terms, You have to derive them seperetely using the rule of differentiation.

![derivative of the objective function with respect to the regularizer][derivative_regularizer]

![derivative of the objective function with respect to the hinge loss function][derivative_loss]

#### TODO:
- plot whatever possible
- take screenshots and upload them to the README.md so that it looks way more professional
- find some eye-catching graphic for the beggining that will be dissplayed in the whole series
---
##### License:
-- MIT

##### **Created by devmood**

LinkedIn: [Albert Millert](https://www.linkedin.com/in/albert-millert/)

Instagram: [devmood](https://instagram.com/devmood/)

Codepen: [devmood](https://codepen.io/devmood/)

[line]: http://www.sciweavers.org/tex2img.php?eq=y%20%3D%20ax%20%2B%20b&bc=White&fc=Black&im=png&fs=12&ff=mathpazo&edit=0
[sse]: http://www.sciweavers.org/tex2img.php?eq=%20SSE%20%3D%20%5Csum%20%28y%20-%20%20%5Chat%7By%7D%20%29%5E2&bc=White&fc=Black&im=png&fs=12&ff=mathpazo&edit=0
[derivative_a]: http://www.sciweavers.org/tex2img.php?eq=%20%5Cfrac%7B%5Cpartial%5Ef%7D%7B%5Cpartial%20a%7D%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum%20-2x%20%28y%20-%20%28ax%20%2B%20b%29%29&bc=White&fc=Black&im=png&fs=12&ff=mathpazo&edit=0
[derivative_b]: http://www.sciweavers.org/tex2img.php?eq=%20%5Cfrac%7B%5Cpartial%5Ef%7D%7B%5Cpartial%20b%7D%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum%20-2%20%28y%20-%20%28ax%20%2B%20b%29%29&bc=White&fc=Black&im=png&fs=12&ff=mathpazo&edit=0
[hlf]: http://www.sciweavers.org/tex2img.php?eq=c%28x%2C%20y%2C%20%20%5Chat%7By%7D%29%20%3D%20%281%20-%20y%20%2A%20%5Chat%7By%7D%29_%2B%20&bc=White&fc=Black&im=png&fs=12&ff=mathpazo&edit=0
[objf]: http://www.sciweavers.org/tex2img.php?eq=min_w%20%5Clambda%20%5Cparallel%20w%20%5Cparallel%5E2%20%2B%20%20%5Csum_%7Bi%3D1%7D%5En%20%281%20-%20y_i%20%2A%20%5Cwidehat%7By_i%7D%29_%2B%20&bc=White&fc=Black&im=png&fs=12&ff=mathpazo&edit=0
[derivative_regularizer]: http://www.sciweavers.org/tex2img.php?eq=%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20w_k%7D%20%5Clambda%20%5Cparallel%20w%20%5Cparallel%5E2%20%3D%202%20%5Clambda%20w_k%20&bc=White&fc=Black&im=png&fs=12&ff=mathpazo&edit=0
[derivative_loss]: http://www.sciweavers.org/tex2img.php?eq=%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20w_k%7D%20%281%20-%20y_i%20%2A%20%5Cwidehat%7By_i%7D%29_%2B%20%3D%20%20%0A%5Cbegin%7Bcases%7D%0A%5C%280%2C%20%26%20y_i%20%2A%20%5Cwidehat%7By_i%7D%20%5Cgeq%201%20%5C%5C%0A%5C-y_i%20x_%7Bik%7D%2C%20%26%20else%0A%5Cend%7Bcases%7D&bc=White&fc=Black&im=png&fs=12&ff=mathpazo&edit=0
