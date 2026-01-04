# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

import activation_functions as act_func

# create input border
p = np.linspace(-2, 2, 100)

# create figure with 5 subplot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('figure for question 2.5', fontsize=16, fontweight='bold')

# i. w=1, b=1, f=hardlims
n1 = 1 * p + 1
a1 = act_func.hardlims(n1)

axes[0, 0].plot(p, a1, 'b-', linewidth=2)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlabel('p')
axes[0, 0].set_ylabel('a')
axes[0, 0].set_title('i. w=1, b=1, f=hardlims')
axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)

# ii. w=-1, b=1, f=hardlims
n2 = -1 * p + 1
a2 = act_func.hardlims(n2)
axes[0, 1].plot(p, a2, 'r-', linewidth=2)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlabel('p')
axes[0, 1].set_ylabel('a')
axes[0, 1].set_title('ii. w=-1, b=1, f=hardlims')
axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)

# iii. w=2, b=3, f=pureline
n3 = 2 * p + 3
a3 = act_func.pureline(n3)
axes[0, 2].plot(p, a3, 'g-', linewidth=2)
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_xlabel('p')
axes[0, 2].set_ylabel('a')
axes[0, 2].set_title('iii. w=2, b=3, f=purelin')
axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 2].axvline(x=0, color='k', linestyle='--', alpha=0.3)

# iv. w=2, b=3, f=satlins
n4 = 2 * p + 3
a4 = act_func.satlins(n4)
axes[1, 0].plot(p, a4, 'm-', linewidth=2)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlabel('p')
axes[1, 0].set_ylabel('a')
axes[1, 0].set_title('iv. w=2, b=3, f=satlins')
axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)

# v. w=-2, b=-1, f=poslin
n5 = -2 * p + (-1)
a5 = act_func.poslin(n5)
axes[1, 1].plot(p, a5, 'c-', linewidth=2)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlabel('p')
axes[1, 1].set_ylabel('a')
axes[1, 1].set_title('v. w=-2, b=-1, f=poslin')
axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)

# delete empty subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('./e2_5_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("figure created successfully!")
