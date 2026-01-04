# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import activation_functions as act_func

# network parameters
W = np.array([[1, 1], 
              [-1, 1]])
b = np.array([[-2], 
              [0]])


# create the points for drawing
p1 = np.linspace(-3, 3, 300)
p2 = np.linspace(-3, 3, 300)
P1, P2 = np.meshgrid(p1, p2)

# calculate the output for each point
n1 = W[0, 0] * P1 + W[0, 1] * P2 + b[0, 0]
n2 = W[1, 0] * P1 + W[1, 1] * P2 + b[1, 0]

a1 = act_func.hardlims(n1)
a2 = act_func.hardlims(n2)

# add color the labels

class_label = np.zeros_like(a1)
class_label[(a1 == -1) & (a2 == -1)] = 0
class_label[(a1 == -1) & (a2 == 1)] = 1
class_label[(a1 == 1) & (a2 == -1)] = 2
class_label[(a1 == 1) & (a2 == 1)] = 3

# drawing the figure
fig, ax = plt.subplots(1, 1, figsize=(10, 10))


contour = ax.contourf(P1, P2, class_label, levels=[-.5, .5, 1.5, 2.5, 3.5], 
                      colors=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'],
                      alpha=0.6)

# drawing boundry decisions
ax.contour(P1, P2, n1, levels=[0], colors='blue', linewidths=2, 
          linestyles='--', label='n₁=0 (p₁+p₂-2=0)')
ax.contour(P1, P2, n2, levels=[0], colors='red', linewidths=2, 
          linestyles='--', label='n₂=0 (-p₁+p₂=0)')

#calculate output for special input
p_test = np.array([[1], [-1]])
n_test = W @ p_test + b
a_test = act_func.hardlims(n_test)


# drawing the test point for verification
ax.plot(p_test[0, 0], p_test[1, 0], 'ko', markersize=15, 
       markerfacecolor='yellow', markeredgewidth=3, 
       label=f' test point: p=[{p_test[0,0]}, {p_test[1,0]}]')

# adding class labels
ax.text(-2, -2, 'a=[-1, -1]', fontsize=14, fontweight='bold', 
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax.text(-2, 2, 'a=[-1, +1]', fontsize=14, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
ax.text(2, -1, 'a=[+1, -1]', fontsize=14, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.text(1.5, 2, 'a=[+1, +1]', fontsize=14, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

ax.set_xlabel('p₁', fontsize=14)
ax.set_ylabel('p₂', fontsize=14)
ax.set_title('decision region for perceptron', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc='upper right')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)

plt.tight_layout()
plt.savefig('./e3_4_decision_regions.png', dpi=300, bbox_inches='tight')
plt.show()

# drawing 3-d figure for each output
fig = plt.figure(figsize=(16, 6))


ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(P1, P2, a1, cmap='viridis', alpha=0.8, 
                         linewidth=0, antialiased=True)
ax1.set_xlabel('p₁')
ax1.set_ylabel('p₂')
ax1.set_zlabel('a₁')
ax1.set_title('first output of neuron (a₁)')
ax1.set_zlim(-1.5, 1.5)
fig.colorbar(surf1, ax=ax1, shrink=0.5)


ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(P1, P2, a2, cmap='plasma', alpha=0.8,
                         linewidth=0, antialiased=True)
ax2.set_xlabel('p₁')
ax2.set_ylabel('p₂')
ax2.set_zlabel('a₂')
ax2.set_title('second output of neuron(a₂)')
ax2.set_zlim(-1.5, 1.5)
fig.colorbar(surf2, ax=ax2, shrink=0.5)

plt.tight_layout()
plt.savefig('./e3_4_3d_outputs.png', dpi=300, bbox_inches='tight')
plt.show()
