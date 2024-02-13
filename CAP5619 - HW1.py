import numpy as np
import matplotlib.pyplot as plt
from numpy import exp as exp
import seaborn as sns
sns.set_theme()

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
    
#%%

import numpy as np
import tensorflow as tf 
from tensorflow import keras
import keras.optimizers
from keras.layers import Dense
from keras.models import Sequential
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import statistics
from keract import display_activations, get_activations, display_heatmaps
#%%

# question 2 part a

model = Sequential()
model.add(Dense(3, input_shape=(1,),activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='linear'))
adam = tf.optimizers.Adam(learning_rate=0.005)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_absolute_error'])
model.summary()

x = np.arange(-2,2, 0.02) # 200 training samples
y = np.sin((math.pi/4)*x)
model.fit(x, y, epochs=1000, batch_size=32, verbose=0)
history = model.history

plt.plot(history.history['mean_absolute_error'])
plt.title('Error vs Epoch')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()

y_hat = model.predict(x).flatten()
max_diff = max(np.abs(y - y_hat))
avg_diff = np.abs(y - y_hat).mean()

plt.plot(x,y, label='sin((πx/4))')
plt.plot(x,y_hat, label='Trained Model')
plt.legend()
plt.annotate('Max Error: {}'.format(round(max_diff,3)), xy=(-2,.25))
plt.annotate('Avg Error: {}'.format(round(avg_diff,3)), xy=(-2,0))
plt.show()


#%%

# question 2 part b

layer_names = [model.layers[i].name for i in range(len(model.layers))][:-1]
layer_1_activations = np.zeros((200,3))
layer_2_activations = np.zeros((200,2))
for i in range(len(x)):
    # Select a single test example
    keract_input = x.reshape(-1,1)[np.newaxis, i, :] # Add a new axis to make the input shape (1, input_shape)
    
    # Get activations for a single test example
    activations = get_activations(model, keract_input, layer_names=layer_names)
    layer_1_activations[i] = activations[layer_names[0]][0]
    layer_2_activations[i] = activations[layer_names[1]][0]

layer_1_activations[layer_1_activations > 0] = 1
layer_2_activations[layer_2_activations > 0] = 1

def find_activation_region(layer_activations):
    region_bounds = []
    for i in range(1,layer_activations.shape[0]):
        if np.array_equal(layer_activations[i], layer_activations[i-1]) is False:
            region_bounds.append(i)
    return region_bounds

layer_1_region_bounds = find_activation_region(layer_1_activations)
layer_2_region_bounds = find_activation_region(layer_2_activations)
region_bounds = sorted(layer_1_region_bounds + layer_2_region_bounds)
# activation_patterns = np.zeros()
#%%
activation_patterns = np.zeros((len(region_bounds)+1,5))
for i in range(len(region_bounds)+1):
    if i < len(region_bounds):
        layer_1_pattern = layer_1_activations[region_bounds[i]-1].astype(int)
        layer_2_pattern = layer_2_activations[region_bounds[i]-1].astype(int)
    else: 
        layer_1_pattern = layer_1_activations[-1].astype(int)
        layer_2_pattern = layer_2_activations[-1].astype(int)
    activation_patterns[i] = np.concatenate((layer_1_pattern,layer_2_pattern))

#%%
for i in range(len(region_bounds)+1):
    if i == 0:
        plt.plot(x[0:region_bounds[i]],y_hat[0:region_bounds[i]], label = 'Activation Pattern: {}'.format(activation_patterns[i]))
    elif i < len(region_bounds):
        plt.plot(x[region_bounds[i-1]:region_bounds[i]], y_hat[region_bounds[i-1]:region_bounds[i]], label = 'Activation Pattern: {}'.format(activation_patterns[i]))
    elif i == len(region_bounds):
        plt.plot(x[region_bounds[i-1]:],y_hat[region_bounds[i-1]:], label = 'Activation Pattern: {}'.format(activation_patterns[i]))
plt.legend()
plt.title('Activation Regions for Trained Model')
    
    


#%%




