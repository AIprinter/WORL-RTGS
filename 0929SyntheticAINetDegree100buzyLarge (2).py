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

WORL_RTGS = {20: 3029.47, 21: 4037.58, 22: 4196.09, 23: 5311.99, 24: 5146.27,
    25: 6173.69, 26: 7033.54
}
DRAS = {20: 3064.06, 21: 4170.68, 22: 4261.96, 23: 5640.78, 24: 5525.92,
    25: 6222.92, 26: 6803.78
}

mDRL = {20: 3143.85, 21: 4094.34, 22: 4365.62, 23: 6070.0, 24: 5275.41,
    25: 6360.16, 26: 7532.66
}

MAGRL = {20: 3533.71, 21: 4407.01, 22: 4761.93, 23: 6555.35, 24: 5775.17,
    25: 7018.19, 26: 8354.81
}

MIGMPS = {20: 3405.79, 21: 4694.46, 22: 4884.25, 23: 6057.59, 24: 5808.52,
    25: 7034.67, 26: 8174.88
}

RL_Hybrid = {20: 3615.87, 21: 4728.1, 22: 5390.76, 23: 7672.01, 24: 6150.95,
    25: 7590.49, 26: 9262.25
}


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
