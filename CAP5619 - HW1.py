import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# relu = lambda x: x if x >= 0 else 0
# sigmoid = lambda x: (1+np.exp(-x))**-1
# pwlu = lambda x: .1*x+.9 if x > 1 else x if -1 <= x <= 1 else .1*x-.9
# swish = lambda x: x*sigmoid(2.5*x)
# elu = lambda x: x if x >= 0 else .05*(np.exp(x)-1)

# x = np.linspace(-5,5,101)
# plt.plot(x, np.gradient([relu(i) for i in x]))
# plt.plot(x, np.gradient([sigmoid(i) for i in x]))
# plt.plot(x, np.gradient([pwlu(i) for i in x]))
# plt.plot(x, np.gradient([swish(i) for i in x]))
# plt.plot(x, np.gradient([elu(i) for i in x]))

def relu_grad(x):
    # x: an array of input values
    out = x
    out[out >= 0] = 1
    out[out < 0] = 0
    return out

input = np.array([-1,0,1,2])

print(relu_grad(input))

def sigmoid_grad(x):
    return np.array([np.exp(-i)/((1 + np.exp(-i))**2) for i in x])

input = np.linspace(-5,5,101)
plt.plot(input,sigmoid_grad(input))
    