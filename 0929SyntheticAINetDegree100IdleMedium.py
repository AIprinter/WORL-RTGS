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

WORL_RTGS = {11: 925.5, 12: 1281.13, 13: 1175.99, 
  14: 1393.09, 15: 1646.4, 16: 1592.22, 17: 1652.92, 18: 2004.97, 
  19: 1976.23, 20: 2230.11}

DRAS = {11: 983.26, 12: 1346.97, 13: 1370.51,
  14: 1414.17, 15: 1680.81, 16: 1700.51, 17: 1808.1, 18: 2033.64,
  19: 2115.18, 20: 2333.58}

mDRL = {11: 1002.02, 12: 1318.65, 13: 1444.27,
  14: 1508.27, 15: 1693.34, 16: 1772.12, 17: 1817, 18: 2115.87,
  19: 2182.22, 20: 2341.52}

MAGRL = {11: 1121.21, 12: 1450.58, 13: 1501.43,
    14: 1560.93, 15: 1653.86, 16: 2018.14, 17: 1910.9, 18: 2324.17, 19: 2314.92,
    20: 2543.44}

MIGMPS = { 11: 1180.36, 12: 1401.91, 13: 1579.14,
    14: 1688.87, 15: 1863.65, 16: 2101.91, 17: 2025.77, 18: 2247.61, 19: 2377.43,
    20: 2562.21}

RL_Hybrid = { 11: 1261.0, 12: 1469.51, 13: 1602.89,
    14: 1830.56, 15: 1900.05, 16: 2235.14, 17: 2295.26, 18: 2154.12, 19: 2584.57,
    20: 2888.68,}


dras = list(DRAS.values())
mdrl = list(mDRL.values())
magrl = list(MAGRL.values())
rl_hybrid = list(RL_Hybrid.values())
migmps = list(MIGMPS.values())
worl_rtgs = list(WORL_RTGS.values())

x = [1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]


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
