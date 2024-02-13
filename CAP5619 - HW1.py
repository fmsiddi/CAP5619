import numpy as np
import matplotlib.pyplot as plt
from numpy import exp as exp
plt.style.use('seaborn')

# relu = lambda x: x if x >= 0 else 0
# sigmoid = lambda x: (1+np.exp(-x))**-1
# pwlu = lambda x: .1*x+.9 if x > 1 else x if -1 <= x <= 1 else .1*x-.9
# swish = lambda x: x*sigmoid(2.5*x)
# elu = lambda x: x if x >= 0 else .05*(np.exp(x)-1)

# x = np.linspace(-5,5,101)
# plt.scatter(x, np.gradient([relu(i) for i in x]))
# plt.scatter(x, np.gradient([sigmoid(i) for i in x]))
# plt.scatter(x, np.gradient([pwlu(i) for i in x]))
# plt.scatter(x, np.gradient([swish(i) for i in x]))
# plt.scatter(x, np.gradient([elu(i) for i in x]))

def relu_grad(x):
    return np.array([1 if i >= 0 else 0 for i in x])

def sigmoid_grad(x):
    return np.array([exp(-i)/((1 + exp(-i))**2) for i in x])
    
def pwlu_grad(x):
    return np.array([1 if abs(i) <= 1 else .1 for i in x])

def swish_grad(x):
    return np.array([exp(2.5*i) * (exp(2.5*i) + 5*i + 2)/(2*(exp(2.5*i) + 1)**2) for i in x])

def elu_grad(x):
    return np.array([1 if i >= 0 else .05*exp(i) for i in x])

def plot_gradient(x, grad_func, title):
    out = grad_func(x)
    inactive = np.where(np.abs(out) == 0.0)[0]
    slow = np.where((np.abs(out) > 0) & (np.abs(out) <= 0.1))[0]
    active = np.where((np.abs(out) >= 0.1) & (np.abs(out) <= 0.99))[0]
    fast = np.where(np.abs(out) > 0.99)[0]
    out = np.array(out)
    plt.scatter(x[inactive], out[inactive], label='Inactive_learning', s=5)
    plt.scatter(x[slow], out[slow], label='Slow_learning', s=5)
    plt.scatter(x[active], out[active], label='Active_learning', s=5)
    plt.scatter(x[fast], out[fast], label='Fast_learning', s=5)
    plt.ylabel("Gradient Value")
    plt.xlabel("Input Value")
    plt.title(title)
    plt.legend(loc='best')
    plt.show()
    
if __name__ == "__main__":
    x = np.linspace(-5,5,1001)
    plot_gradient(x, relu_grad, 'Gradient of ReLU')
    plot_gradient(x, sigmoid_grad, 'Gradient of Sigmoid')
    plot_gradient(x, pwlu_grad, 'Gradient of PwLU')
    plot_gradient(x, swish_grad, 'Gradient of Swish')
    plot_gradient(x, elu_grad, 'Gradient of ELU')