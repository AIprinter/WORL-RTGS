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
DRAS = {9: 420, 10: 318, 11: 381, 12: 379, 13: 357, 14: 604, 15: 320, 16: 687, 17: 955, 18: 949}
mDRL = {9: 450, 10: 395, 11: 462, 12: 395, 13: 432, 14: 655, 15: 375, 16: 737, 17: 725, 18: 955}
MAGRL = {9: 510, 10: 495, 11: 505, 12: 525, 13: 495, 14: 755, 15: 565, 16: 865, 17: 825, 18: 1095}
RL_Hybrid = {9: 680, 10: 485, 11: 695, 12: 675, 13: 705, 14: 925, 15: 675, 16: 965, 17: 931, 18: 1355}
MIGMPS = {9: 560, 10: 475, 11: 525, 12: 565, 13: 540, 14: 785, 15: 480, 16: 965, 17: 855, 18: 1135}

WORL_RTGS = {9: 391, 10: 255, 11: 325, 12: 321, 13: 308, 14: 533, 15: 260, 16: 622, 17: 612, 18: 891}


dras = list(DRAS.values())
mdrl = list(mDRL.values())
magrl = list(MAGRL.values())
rl_hybrid = list(RL_Hybrid.values())
migmps = list(MIGMPS.values())
worl_rtgs = list(WORL_RTGS.values())

x = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]


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
