# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of GTMR (Generative Topographic Mapping Regression)

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from sklearn.datasets.samples_generator import make_swiss_roll
import mpl_toolkits.mplot3d

from gtm import gtm

# settings
shape_of_map = [30, 30]
shape_of_rbf_centers = [4, 4]
variance_of_rbfs = 0.5
lambda_in_em_algorithm = 0.001
number_of_iterations = 300
display_flag = 1
number_of_samples = 1000
noise_ratio_of_y = 0.1
random_state_number = 30000

# load a swiss roll dataset and make a y-variable
original_X, color = make_swiss_roll(number_of_samples, 0, random_state=10)
X = original_X
raw_y = 0.3 * original_X[:, 0] - 0.1 * original_X[:, 1] + 0.2 * original_X[:, 2]
original_y = raw_y + noise_ratio_of_y * raw_y.std(ddof=1) * np.random.randn(len(raw_y))

# plot
plt.rcParams["font.size"] = 18
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(original_X[:, 0], original_X[:, 1], original_X[:, 2], c=original_y)
fig.colorbar(p)
plt.show()

# divide a dataset into training data and test data
Xtrain = original_X[:500, :]
ytrain = original_y[:500]
Xtest = original_X[500:, :]
ytest = original_y[500:]

# autoscaling
# autoscaled_X = (original_X - original_X.mean(axis=0)) / original_X.std(axis=0,ddof=1)
autoscaled_Xtrain = (Xtrain - Xtrain.mean(axis=0)) / Xtrain.std(axis=0, ddof=1)
autoscaled_Xtest = (Xtest - X.mean(axis=0)) / X.std(axis=0, ddof=1)
autoscaled_ytrain = (ytrain - ytrain.mean()) / ytrain.std(ddof=1)

# construct GTMR model
input_dataset = np.c_[autoscaled_Xtrain, autoscaled_ytrain]
model = gtm(shape_of_map, shape_of_rbf_centers, variance_of_rbfs, lambda_in_em_algorithm, number_of_iterations,
            display_flag)
model.fit(input_dataset)
if model.success_flag:
    # calculate of responsibilities
    responsibilities = model.responsibility(input_dataset)

    # plot the mean of responsibilities
    means = responsibilities.dot(model.map_grids)
    plt.figure()
    #    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(means[:, 0], means[:, 1], c=ytrain)
    plt.colorbar()
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.xlabel("z1 (mean)")
    plt.ylabel("z2 (mean)")
    plt.show()

    # plot the mode of responsibilities
    modes = model.map_grids[responsibilities.argmax(axis=1), :]
    plt.figure()
    #    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(modes[:, 0], modes[:, 1], c=ytrain)
    plt.colorbar()
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.xlabel("z1 (mode)")
    plt.ylabel("z2 (mode)")
    plt.show()

# GTMR prediction
predicted_ytest_mean, predicted_ytest_mode, responsibilities, px = model.gtmr_predict(autoscaled_Xtest)
predicted_ytest = predicted_ytest_mean * ytrain.std(ddof=1) + ytrain.mean()  # mean
# predicted_ytest = predicted_ytest_mode * ytrain.std(ddof=1) + ytrain.mean() # mode
predicted_ytest = np.ndarray.flatten(predicted_ytest)

# r2p, RMSEp, MAEp
print("r2p: {0}".format(float(1 - sum((ytest - predicted_ytest) ** 2) / sum((ytest - ytest.mean()) ** 2))))
print("RMSEp: {0}".format(float((sum((ytest - predicted_ytest) ** 2) / len(ytest)) ** (1 / 2))))
print("MAEp: {0}".format(float(sum(abs(ytest - predicted_ytest)) / len(ytest))))
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(ytest, predicted_ytest)
YMax = np.max(np.array([np.array(ytest), predicted_ytest]))
YMin = np.min(np.array([np.array(ytest), predicted_ytest]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel("simulated y")
plt.ylabel("estimated y")
plt.show()
