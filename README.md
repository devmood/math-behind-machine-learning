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
