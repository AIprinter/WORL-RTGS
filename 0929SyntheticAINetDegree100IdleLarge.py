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
21: 3003.35, 22: 3548.9, 23: 3715.88, 
  24: 4902.64, 25: 4661.22, 26: 5236.25, 27: 5626.03}

DRAS = {
21: 3136.8, 22: 3860.33, 23: 4244.46,
  24: 5061.87, 25: 4770.09, 26: 5688.4, 27: 6611.01}

mDRL = {
21: 3256.16, 22: 4360.44, 23: 4682.12,
  24: 5240.64, 25: 5081.95, 26: 5944.54, 27: 6724.02
}

MAGRL = {21: 3228.11, 22: 4296.24, 23: 4561.1, 24: 5979.81, 25: 5386.66,
    26: 6411.6, 27: 7584.14}

MIGMPS = {21: 3260.03, 22: 4333.27, 23: 4434.19, 24: 5830.02, 25: 5480.09,
    26: 6639.01, 27: 7728.71
}

RL_Hybrid = { 21: 3323.42, 22: 4395.36, 23: 4578.25, 24: 6171.46, 25: 5622.62,
    26: 6890.06, 27: 8050.78}


dras = list(DRAS.values())
mdrl = list(mDRL.values())
magrl = list(MAGRL.values())
rl_hybrid = list(RL_Hybrid.values())
migmps = list(MIGMPS.values())
worl_rtgs = list(WORL_RTGS.values())

x = [4000, 5000, 6000, 7000, 8000, 9000, 10000]


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
