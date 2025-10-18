#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   demo50.py

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2025-09-12 8:28   nana      1.0         highest bandwidth is 2000
'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42   # 输出 PDF 使用 Type 42 (TrueType)
matplotlib.rcParams['ps.fonttype'] = 42    # 输出 PS 使用 Type 42

WORL_RTGS = {1: 105.85, 2: 251.92, 3: 315.4, 4: 354.67, 5: 453.36, 6: 532.61, 7: 651.35,
  8: 745.91, 9: 726.45, 10: 886.15}

DRAS = {1: 115.23, 2: 260.35, 3: 360.6, 4: 395.63, 5: 507.57, 6: 654.55, 7: 671.9,
  8: 797.7, 9: 745.45, 10: 964.99}

mDRL = {1: 117.52, 2: 264.82, 3: 385.76, 4: 465.29, 5: 516.71, 6: 714.01, 7: 697.44,
  8: 826.71, 9: 780.28, 10: 970.54}

MAGRL = {1: 123.4, 2: 295.75, 3: 369.05, 4: 507.27, 5: 630.06, 6: 667.11, 7: 752.34,
    8: 926.19, 9: 884.04, 10: 1053.63}

MIGMPS = {1: 125.61, 2: 309.17, 3: 408.36, 4: 525.06, 5: 669.96, 6: 732.94, 7: 776.2,
    8: 956.04, 9: 867.18, 10: 1046.2}

RL_Hybrid = {1: 137.57, 2: 313.48, 3: 431.22, 4: 594.04, 5: 701.39, 6: 726.74, 7: 793.49,
    8: 1009.64, 9: 987.85, 10: 1197.87}


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
