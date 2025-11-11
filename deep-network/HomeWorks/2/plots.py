# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 08:45:33 2024

@author: masoo
"""

############example1#########
def hardlim(x):
    if x<0:
        return 0
    else:
        return 1

import numpy as np
p = np.linspace(-2, 2, 20)
a = [hardlim(i-1) for i in p]
# a = np.sign(p+1)
from  matplotlib import pyplot as plt
plt.plot(p,a)


######example2##############
def np_hardlim(x):
    x[x<0]=0
    x[x!=0]=1
    return x
p1 = np.linspace(-2, 2, 20)
p2 = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(p1, p2)
n = X + Y -2
a = np_hardlim(n)


# fig = plt.figure(figsize=plt.figaspect(2.))
fig = plt.figure()
# fig.suptitle('A tale of 2 subplots')

# First subplot
ax = fig.add_subplot(1, 1, 1)

ax = fig.add_subplot(1, 1, 1, projection='3d')

surf = ax.plot_surface(X, Y, a,
                        linewidth=0, antialiased=False)
ax.set_zlim(-0.5, 1.2)

plt.show()

