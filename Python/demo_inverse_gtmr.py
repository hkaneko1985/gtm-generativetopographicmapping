# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of inverse GTMR (Generative Topographic Mapping Regression)

import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from sklearn.datasets.samples_generator import make_swiss_roll
import matplotlib.figure as figure
import mpl_toolkits.mplot3d

from gtm import gtm

target_y_value = 4  # y-target for inverse analysis

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

# autoscaling
autoscaled_X = (original_X - original_X.mean(axis=0)) / original_X.std(axis=0, ddof=1)
autoscaled_y = (original_y - original_y.mean()) / original_y.std(ddof=1)

# construct GTMR model
input_dataset = np.c_[autoscaled_X, autoscaled_y]
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
    plt.scatter(means[:, 0], means[:, 1], c=original_y)
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
    plt.scatter(modes[:, 0], modes[:, 1], c=original_y)
    plt.colorbar()
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.xlabel("z1 (mode)")
    plt.ylabel("z2 (mode)")
    plt.show()

# inverse analysis
estimated_x_mean, estimated_x_mode, responsibilities_inverse, py = model.inverse_gtmr(target_y_value)
estimated_x_mean = estimated_x_mean * original_X.std(axis=0, ddof=1) + original_X.mean(axis=0)
estimated_x_mode = estimated_x_mode * original_X.std(axis=0, ddof=1) + original_X.mean(axis=0)
# print("estimated x-mean: {0}".format(estimated_x_mean))
print("estimated x-mode: {0}".format(estimated_x_mode))

estimated_x_mean_on_map = responsibilities_inverse.dot(model.map_grids)
estimated_x_mode_on_map = model.map_grids[np.argmax(responsibilities_inverse), :]
# print("estimated x-mean on map: {0}".format(estimated_x_mean_on_map))
print("estimated x-mode on map: {0}".format(estimated_x_mode_on_map))
