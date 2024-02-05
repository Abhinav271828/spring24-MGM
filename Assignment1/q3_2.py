# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
s1 = [1,0,1,0,0,1,1,0,0,1]
s2 = [0,0,1,1,0,1,0,1,0,0]
s3 = [1,1,0,1,0,0,0,0,0,0]
s4 = [0,0,0,1,0,1,0,0,1,1]
s5 = [0,0,0,0,0,0,1,0,1,1]
s6 = [1,1,1,0,0,1,0,1,0,0]
s7 = [1,0,1,0,1,0,1,1,1,1]
s8 = [0,0,0,0,1,1,0,1,0,0]
s9 = [0,0,0,1,0,0,1,0,1,1]
s10 = [1,1,0,0,1,1,0,1,1,1]
s11 = [0,1,0,0,1,0,1,1,1,1]
s12 = [0,1,1,1,1,1,1,0,1,0]

samples = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12]

theta1 = None
theta2 = [None, None]
#theta2[k] = probability of 1 given x_1 = k
thetas = [theta1, theta2] + \
         [[None, None, None, None] for _ in range(3, 11)]
#thetai[k] = probability of 1 given x_{i-1} = k//2 and x_{i-2} = k % 2

thetas[0] = sum(1 for s in samples if s[0] == 1)/12

for k in range(2):
    thetas[1][k] = sum(1 for s in samples if s[1] == 1 and s[0] == k)/sum(1 for s in samples if s[0] == k)

for i in range(2, 10):
    for k in range(4):
        thetas[i][k] = sum(1 for s in samples if s[i] == 1 and s[i-1] == k//2 and s[i-2] == k%2)/(sum(1 for s in samples if s[i-1] == k//2 and s[i-2] == k%2))

# %%
def gen_sample():
    first = np.random.binomial(n=1, p=thetas[0])
    second = np.random.binomial(n=1, p=thetas[1][first])
    sample = [first, second]
    for theta in thetas[2:]:
        ith = np.random.binomial(n=1, p=theta[sample[-1]*2 + sample[-2]])
        sample.append(ith)
    return sample

# %%
for _ in range(10):
    print(gen_sample())


