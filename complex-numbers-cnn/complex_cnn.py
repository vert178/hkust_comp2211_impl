# -*- coding: utf-8 -*-
"""
Step 1: Import the required libraries
"""

import os
# Supress tensor flow warnings since it is a bit annoying
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, RandomRotation, RandomZoom, Dense, Dropout, Flatten, Rescaling
import matplotlib.pyplot as plt
from keras.initializers import HeNormal, HeUniform

"""Step 2: Generate the required data"""

# Utility stuff
lpl = 0
def fprint(fr, total_fr, description, print_every = 8, len_bar = 24):
		global lpl
		ratio = round((fr + 1)/total_fr * len_bar)
		st = description+ ": [" + ratio * "=" + (len_bar - ratio) * " " + "]  " + str(fr + 1) + "/" + str(total_fr)
		if fr & (2 ** print_every - 1) == 0:
			print("\b" * lpl + st, end = "", flush = True)
		lpl = len(st)

def complex_to_rgb(z: np.ndarray):
    # Use broadcasting magic to convert the grid into the right shape
    res = np.zeros(z.shape + (3,))

    # 0 = H, 1 = S, 2 = V
    # Change to degree and normalize to [0, 179]
    res[:, : ,0] = np.angle(z, deg = True) + 180
    res[:, : ,0] *= 180 / 360

    __sat_cut = (1, 1)
    mag = np.abs(z)
    res[:, :, 1][mag != 0] = __sat_cut[1] / mag[mag != 0] * 255
    res[:, :, 1][mag < __sat_cut[1]] = 255
    
    res[:, :, 2]  = mag / __sat_cut[0] * 255
    res[:, :, 2][mag >= __sat_cut[0]] = 255

    result = np.array(res, dtype = np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_HSV2RGB)

def fac(n: int):
    if n == 0: return 1
    return n * fac(n-1)

dimensions = (100, 100)
powers = (20, 5)

def generate_powers(z, max_power):
    cum_z = np.ones((1,) + z.shape, dtype = z.dtype)
    z_powers = np.ones((1,) + z.shape, dtype = z.dtype)
    for i in range(max_power - 1):
        cum_z = np.multiply(cum_z, z)
        cz = np.array(cum_z, dtype = cum_z.dtype)
        cz /= fac(i)
        z_powers = np.concatenate((z_powers, cum_z), axis = 0)
    return z_powers

def generate_z_power(xy_range = (-1, 1, -1, 1)):
    x_min, x_max, y_min, y_max = xy_range
    x_plot = x_min + (x_max - x_min) / dimensions[1] * np.arange(dimensions[1])
    y_plot = (y_max + (y_min - y_max) / dimensions[0] * np.arange(dimensions[0]))
    y_plot = y_plot.reshape(-1, 1) * 1j
    z = x_plot + y_plot
    z_bar = x_plot - y_plot

    z_powers = generate_powers(z, powers[0])
    z_bar_powers = generate_powers(z_bar, powers[1])
    return z_powers, z_bar_powers

z_power, z_bar_power = generate_z_power()

def generate_coefficients(size) -> np.ndarray:
    return np.random.normal(0, 1, size)

def generate_data(is_holomorphic):
    coefficients = generate_coefficients(powers[0])
    z_powers = np.array(z_power, dtype = z_power.dtype).transpose(1,0,2)
    z_bar_powers = np.array(z_bar_power, dtype = z_power.dtype).transpose(1,0,2)
    if is_holomorphic:
        haha = np.dot(coefficients, z_powers)
    else:
        coefficients[1 : powers[1]] *= 0.5
        coefficients2 = generate_coefficients(powers[1] - 1)
        haha = np.dot(coefficients, z_powers) + np.dot(coefficients2, z_bar_powers[:, 1:])
    return haha

def generate_datas(count = 50000):
    X = np.zeros((count,) + dimensions + (3,), dtype = np.uint8)
    y = np.zeros((count,), dtype = np.uint8)
    for i in range(count):
        is_holomorphic = np.random.random() > 0.5
        data = generate_data(is_holomorphic)
        data = complex_to_rgb(data)
        X[i] = data
        y[i] = is_holomorphic
        fprint(i, count, "Generated", 8)
    print()
    return X, y

X_train, y_train = generate_datas(count = 60000)
X_test, y_test = generate_datas(count = 10000)

"""Step 3: Explore the data"""

i = 5
plt.figure()
plt.imshow(X_train[i])
print(y_train[i] == 1)

j = 0
plt.figure()
plt.imshow(X_train[j])
print(y_train[j] == 1)

cum = []
for k in range(255):
    a = np.count_nonzero(np.abs(X_train[i] - X_train[j]) > k)
    cum.append([a])
plt.figure()
plt.plot(cum)

"""Step 4-7: Build, compile, train and save the model"""

def create_model():
    model = Sequential()
    model.add(Rescaling(1./255, input_shape=(100, 100, 3)))
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(320, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

class BCP(keras.callbacks.Callback):
		accuracy = []
		loss = []

		def __init__(self):
				super(BCP, self).__init__()

		def on_train_batch_end(self, batch, logs=None):                                
				BCP.accuracy.append(logs.get('accuracy'))
				BCP.loss.append(logs.get('loss'))
		
		# Plot everything
		@staticmethod
		def plot(history):
			plt.figure()
			plt.plot(BCP.accuracy, color="blue")
			plt.plot(BCP.loss, color="cyan")
			plt.title('model accuracy')
			plt.ylabel('accuracy')
			plt.xlabel('batch')
			plt.legend(['acc', 'loss'], loc='upper left')

			plt.figure()
			plt.plot(history.history['accuracy'], color="blue")
			plt.plot(history.history['val_accuracy'], color="cyan")
			plt.plot(history.history['loss'], color="orange")
			plt.plot(history.history['val_loss'], color="green")
			plt.title('model accuracy')
			plt.ylabel('accuracy')
			plt.xlabel('epoch')
			plt.legend(['acc','valacc', 'loss', 'valloss'], loc='upper left')
			plt.show()
			return

def train_model(model, x_train, y_train, x_test, y_test, batch = 128, epochs = 5, cp_dir = "training_1/cp.ckpt", show_summary = True, save_dir = None):
    keras.utils.set_random_seed(42069)

    model.build(input_shape = (100, 100, 3))

    if show_summary: 
        model.summary()

    # Use checkpoint to prevent losses
    checkpoint_dir = os.path.dirname(cp_dir)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, save_weights_only=True, verbose=1)
    model.compile(optimizer = 'adam', loss = 'MSE', metrics = ['accuracy'])

    # Training
    history = model.fit(x_train, y_train, batch_size = batch, epochs = epochs, validation_data=(x_test, y_test), callbacks=[cp_callback, BCP()])

    # Evaluate the data using x_test and y_test
    score = model.evaluate(x_test, y_test, verbose = 0, batch_size = 128)
    print("Loss Score:", score[0])
    print("Test Accuracy:", score[1])

    # Plot the training results using our BCP class thing
    BCP.plot(history)

    # Save the model
    if save_dir is not None: 
        model.save(save_dir)

    # Return the score which is an array [error, accuracy], training history and the model itself
    return score, history, model

model = create_model()
score, history, model = train_model(model, X_train, y_train, X_test, y_test, batch = 128, epochs = 2, save_dir = "complex_cnn.h5")

"""Step ?? : Play around with the model"""

z = z_power[1]
z_ = z_bar_power[1]

import scipy.special as sp
import time

PI = np.pi


# Helper function for a loading bar-esque progress function
lpl = 0
def fprint(fr, total_fr, func_name):
    global lpl
    len_bar = 24
    ratio = round((fr + 1)/total_fr * len_bar)
    st = func_name + ": [" + ratio * "=" + (len_bar - ratio) * " " + "]  " + str(fr + 1) + "/" + str(total_fr)
    print("\b" * lpl + st, end = "", flush = True)
    lpl = len(st)


# Make sure these functions are numpy array friendly if you decide to implement something

# The identity function also fixes the bullshit values
def id(z):
    z[z.real == np.nan] = np.infty
    z[z.imag == np.nan] = np.infty
    z[z == np.nan] = np.infty
    return z


def three(z):
    return 3 * z


# By default we treat all nan as infinity
def inv(z):
    res = np.zeros_like(z)
    res[z != 0] = 1 / z[z != 0]
    res[z == 0] = np.infty
    return id(res)

def exp(z):
    return np.exp(z)


def sin(z):
    return np.sin(z)


def cos(z):
    return np.cos(z)

def log(z):
    res = np.zeros_like(z)
    res[np.logical_or(z.imag != 0, z.real > 0)] = np.log(z[np.logical_or(z.imag != 0, z.real > 0)])
    return res

def e1z(z):
    return exp(inv(z))


def gamma(z):
    return id(sp.gamma(z))

# True zeta is the one without analytic continuation
def true_zeta(z, k=100):
    res = np.zeros_like(z)
    for i in range(1, k):
        res += inv(exp(z * np.log(i)))
    return res

# The usual Riemann zeta function that we talk about
def zeta(z):
    res = np.zeros_like(z)
    # Main part
    res[z.real > 1] = true_zeta(z[z.real > 1])
    # Analytic continuation
    res[z.real <= 1] = exp(z[z.real <= 1] * np.log(2)) * exp((z[z.real <= 1] - 1) * np.log(PI)) * sin(PI * z[z.real <= 1] / 2) * gamma(1 - z[z.real <= 1]) * true_zeta(1 - z[z.real <= 1])
    return res

# w1 and w2 are the two parameters
# If you want to plot WP, you need a helper function like:
# def Fancy_P(z):
#     return Weierstrass_P(z, 1-3j, 2 + 1j)
def Weierstrass_P(z, w1=2 - 1j, w2=1 + 1j):
    res = inv(z ** 2)
    for i in range(-20, 20):
        for k in range(-20, 20):
            if i == 0 and k == 0:
                continue
            l = i * w1 + k * w2
            res += inv((z - l) ** 2) - 1 / l ** 2
    return res

# Partial sums of zeta
def zeta_partial(z, i):
    res = np.zeros_like(z)
    res[z.real > 1] = inv(exp(z[z.real > 1] * np.log(i)))
    res[z.real <= 1] = exp(z[z.real <= 1] * np.log(2)) * exp((z[z.real <= 1] - 1) * np.log(PI)) * sin(PI * z[z.real <= 1] / 2) * gamma(1 - z[z.real <= 1]) * inv(exp((1-z[z.real <= 1]) * np.log(i)))
    return res


# f is the function and i is the number of iterations
def Newtons_Attractor(z, f, df, iter):
    for _ in range(iter):
        z = z - f(z) * inv(df(z))
    return z

# Eisenstein Series E2k
def ES(z, k, precision = 20, show_progress = True):
    q = exp(2 * PI * 1j * z)
    summ = np.ones_like(q)

    # Calculate E4 using the summation definition
    q_powers = q
    c = round(2 / sp.zeta(1 - 2 * k), 5)
    for n in range(1, precision):
        if show_progress: fprint(n, precision, "Calculating E{}".format(2 * k))
        # assert np.count_nonzero(np.abs(q_powers - q ** n) > 1e-10) == 0
        summ += c * (n ** (2 * k - 1)) * q_powers / (1 - q_powers)
        q_powers = np.multiply(q_powers, q)
    if show_progress: print()
    return summ

def j(z, precision = 100, show_progress = True):
    res = np.zeros_like(z)
    E4t3 = ES(z, 2, precision, show_progress) ** 3
    E6t2 = ES(z, 3, precision, show_progress) ** 2
    res = 1728 * E4t3 * inv(E4t3 - E6t2)
    return res

haha = j(z + 1j)
im = complex_to_rgb(haha)
plt.figure()
plt.imshow(im)
im = im.reshape((1,) + im.shape)
prediction = model.predict(im)
print("Value: ", prediction[0][0], ", is holomorphic? ", prediction[0][0] > 0.2)