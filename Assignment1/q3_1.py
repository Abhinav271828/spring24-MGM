# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
theta_1 = 5/12

theta_2 = [3/5, 2/7]
theta_3 = [2/5, 2/7]
theta_4 = [1/2, 3/8]
theta_5 = [1/5, 4/7]
theta_6 = [3/5, 4/7]
theta_7 = [2/7, 4/5]
theta_8 = [1/3, 2/3]
theta_9 = [1/2, 2/3]
theta_10 = [6/7, 1/5]

thetas = [theta_2, theta_3, theta_4, theta_5, theta_6, theta_7, theta_8, theta_9, theta_10]

# %%
def gen_sample():
    first = np.random.binomial(n=1, p=theta_1)
    sample = [first]
    for theta in thetas:
        ith = np.random.binomial(n=1, p=theta[1-sample[-1]])
        sample.append(ith)
    return sample

# %%
for _ in range(10):
    print(gen_sample())


