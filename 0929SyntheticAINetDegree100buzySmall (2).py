#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   demo50.py

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022-09-12 8:28   nana      1.0         highest bandwidth is 2000
'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42   # 输出 PDF 使用 Type 42 (TrueType)
matplotlib.rcParams['ps.fonttype'] = 42    # 输出 PS 使用 Type 42

WORL_RTGS = {
    0: 119.77, 1: 282.41, 2: 376.36, 3: 470.25, 4: 554.09, 5: 672.46, 6: 701.92,
    7: 853.53, 8: 828.34, 9: 994.92
}
DRAS = {
    0: 120.44, 1: 291.3, 2: 378.06, 3: 483.9, 4: 568.77, 5: 694.38, 6: 724.75,
    7: 870.42, 8: 846.38, 9: 1013.61
}

mDRL = {
    0: 123.01, 1: 295.4, 2: 391.13, 3: 524.98, 4: 604.85, 5: 730.77, 6: 738.48,
    7: 899.81, 8: 852.45, 9: 1014.69
}

MAGRL = {
    0: 130.21, 1: 305.64, 2: 404.5, 3: 569.99, 4: 661.64, 5: 721.39, 6: 786.89,
    7: 952.46, 8: 898.45, 9: 1110.79
}

MIGMPS = {
    0: 127.83, 1: 307.41, 2: 429.46, 3: 554.11, 4: 664.79, 5: 775.37, 6: 774.38,
    7: 998.19, 8: 944.55, 9: 1136.92
}

RL_Hybrid = {
    0: 137.21, 1: 318.06, 2: 445.26, 3: 608.28, 4: 700.18, 5: 792.42, 6: 850.16,
    7: 1082.02, 8: 1004.71, 9: 1178.04
}


dras = list(DRAS.values())
mdrl = list(mDRL.values())
magrl = list(MAGRL.values())
rl_hybrid = list(RL_Hybrid.values())
migmps = list(MIGMPS.values())
worl_rtgs = list(WORL_RTGS.values())

x = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


plt.plot(x, dras, linestyle="--", marker=",", linewidth='1', label="DRAS")
plt.plot(x, mdrl, linestyle="--", marker="+", linewidth='1', label="mDRL")
plt.plot(x, magrl, linestyle="-.", marker="x", linewidth='1', label="MAGRL")
plt.plot(x, rl_hybrid, linestyle="-", marker="v", linewidth='1', label="RL-Hybrid")
plt.plot(x, migmps, linestyle="-.", marker="*", linewidth='1', label="MIG-MPS")
plt.plot(x, worl_rtgs, linestyle="-", marker="o", linewidth='1', label="WORL-RTGS")


plt.xlabel('Nodes number', fontsize=12)
plt.ylabel('Makespan', fontsize=12)

plt.legend(ncol=3, fontsize=10)
plt.show()
