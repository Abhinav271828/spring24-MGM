# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
n = int(input("Enter the value of n: "))
samples = input("Enter k samples: ")
samples = [int(x) for x in samples.split(' ')]

probabilities = np.array([sum(1 for x in samples if x == i)/len(samples) for i in range(1, n+1)])

print("Plotting the histogram!")
np.histogram(probabilities, bins=n)

# %%
def sample(probabilities):
    probabilities = probabilities.cumsum()
    u = np.random.uniform(0, 1)
    if u < probabilities.min(): return 1
    if u > probabilities.max(): return probabilities.shape[0]
    lower = [i for i, p in enumerate(probabilities) if p <= u][-1]
    upper = [i for i, p in enumerate(probabilities) if p >= u][0]
    if (u - probabilities[lower]) < (probabilities[upper] - u):
        return lower+1
    elif (probabilities[upper] - 1) < (u - probabilities[lower]):
        return upper+1
    else:
        np.random.choice([lower+1, upper+1])

# %%
sample(probabilities)


