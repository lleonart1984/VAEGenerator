import VAEDistributor
import random
import math
import numpy as np
import os
import matplotlib.pyplot as plt


plt.scatter([x/10.0 for x in range(0,10)], [x*x/100 for x in range(0,10)], c="g", alpha=0.5, marker='o')
plt.show()

exit()

# Create the model
# We will test generating two variables X,Y 
# where X is a uniform distribution and 
# Y is a gaussian using sin(X*o+p) as mean and q as standard deviation
# parameter_size is 3 (o, p, q)
# sample_size is 2 (X,Y)
# Layers used L=3 
vae = VAEDistributor.VAEGenerator("sinusGaussian", 3, 2, 3)

# Random generator
rnd = random.Random()

def getNewParameter(rnd : random.Random):
    o, p, q = rnd.uniform(1, 5), rnd.uniform(0, 2*3.1416), rnd.uniform(0, 0.3)
    return [o, p,q]

def getNewSample(rnd : random.Random, P):
    o, p, q = P # unroll parameters
    X = rnd.uniform(0, 1)
    Y = rnd.gauss(math.sin(X*o + p), q)
    return [X, Y]

# generating samples
print("generating samples")
N = 10000
Ps = []
Xs = []
for i in range(N):
    P = getNewParameter(rnd)
    Ps.append(P)
    X = getNewSample(rnd, P)
    Xs.append(X)

# train model
print("training model")
vae.trainLoop(Ps, Xs)
vae.saveModel()

# testing model
P = getNewParameter(rnd)
Xs = [] # real samples
for i in range(300): 
    Xs.append(getNewSample(rnd, P))
Xs = np.array(Xs)

gXs = vae.generateSamples(P, 300)



