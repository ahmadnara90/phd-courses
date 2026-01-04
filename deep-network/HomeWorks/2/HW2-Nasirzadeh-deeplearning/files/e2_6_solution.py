# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import activation_functions as act_func


# network parameters
# first layer
w1_11 = 2
w1_21 = 1
b1_1 = 2
b1_2 = -1

# second layer
w2_11 = 1
w2_12 = -1
b2_1 = 0

# create input layer 
p = np.linspace(-3, 3, 200)

# calculate first layer
n1_1 = w1_11 * p + b1_1  
a1_1 = act_func.satlins(n1_1)

n1_2 = w1_21 * p + b1_2
a1_2 = act_func.satlins(n1_2)

# calculate second layer
n2_1 = w2_11 * a1_1 + w2_12 * a1_2 + b2_1  
a2_1 = act_func.pureline(n2_1)

# create the figures
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('figure for question 2_6', fontsize=16, fontweight='bold')

# i. n1^1
axes[0, 0].plot(p, n1_1, 'b-', linewidth=2)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlabel('p')
axes[0, 0].set_ylabel('n₁¹')
axes[0, 0].set_title('i. n₁¹ = 2p + 2')
axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)

# ii. a1^1
axes[0, 1].plot(p, a1_1, 'r-', linewidth=2)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlabel('p')
axes[0, 1].set_ylabel('a₁¹')
axes[0, 1].set_title('ii. a₁¹ = satlins(n₁¹)')
axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)

# iii. n2^1
axes[0, 2].plot(p, n1_2, 'g-', linewidth=2)
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_xlabel('p')
axes[0, 2].set_ylabel('n₂¹')
axes[0, 2].set_title('iii. n₂¹ = p - 1')
axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 2].axvline(x=0, color='k', linestyle='--', alpha=0.3)

# iv. a2^1
axes[1, 0].plot(p, a1_2, 'm-', linewidth=2)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlabel('p')
axes[1, 0].set_ylabel('a₂¹')
axes[1, 0].set_title('iv. a₂¹ = satlins(n₂¹)')
axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)

# v. n1^2
axes[1, 1].plot(p, n2_1, 'c-', linewidth=2)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlabel('p')
axes[1, 1].set_ylabel('n₁²')
axes[1, 1].set_title('v. n₁² = a₁¹ - a₂¹')
axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)

# vi. a1^2
axes[1, 2].plot(p, a2_1, 'orange', linewidth=2)
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_xlabel('p')
axes[1, 2].set_ylabel('a₁²')
axes[1, 2].set_title('vi. a₁² = pureline(n₁²)')
axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 2].axvline(x=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('./e2_6_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("figure created successfully!")
