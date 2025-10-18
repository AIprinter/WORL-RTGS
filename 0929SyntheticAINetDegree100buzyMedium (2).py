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

WORL_RTGS = {10: 1096.86, 11: 1354.39, 12: 1475.65,
    13: 1550.67, 14: 1740.02, 15: 1879.34, 16: 1891.82, 17: 2183.42, 18: 2219.1,
    19: 2418.6
}
DRAS = {10: 1134.07, 11: 1396.76, 12: 1486.93,
    13: 1570.29, 14: 1747.24, 15: 2017.17, 16: 1969.82, 17: 2205.09, 18: 2185.46,
    19: 2480.52
}

mDRL = {10: 1150.9, 11: 1440.82, 12: 1511.98,
    13: 1600.68, 14: 1752.28, 15: 1953.34, 16: 1986.77, 17: 2238.5, 18: 2313.06,
    19: 2536.78
}

MAGRL = {10: 1228.13, 11: 1477.37, 12: 1634.76,
    13: 1774.22, 14: 1806.94, 15: 2103.13, 16: 2166.35, 17: 2372.84, 18: 2520.64,
    19: 2603.66
}

MIGMPS = {10: 1183.65, 11: 1487.39, 12: 1654.32,
    13: 1833.29, 14: 1756.18, 15: 2049.95, 16: 2121.78, 17: 2441.77, 18: 2588.77,
    19: 2633.76
}

RL_Hybrid = {10: 1312.58, 11: 1499.05, 12: 1699.47,
    13: 1868.21, 14: 1817.94, 15: 2138.09, 16: 2251.56, 17: 2525.04, 18: 2740.5,
    19: 2692.17
}


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
