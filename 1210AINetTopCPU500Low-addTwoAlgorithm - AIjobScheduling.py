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
DRAS = {0: 275, 1: 270, 2: 135, 3: 125, 4: 280, 5: 165, 6: 245, 7: 145, 8: 250, 9: 420}
mDRL = {0: 300, 1: 295, 2: 160, 3: 140, 4: 310, 5: 195, 6: 280, 7: 170, 8: 285, 9: 450}
MAGRL = {0: 350, 1: 380, 2: 140, 3: 190, 4: 360, 5: 190, 6: 340, 7: 160, 8: 350, 9: 510}
MIGMPS = {0: 400, 1: 340, 2: 210, 3: 180, 4: 410, 5: 270, 6: 330, 7: 230, 8: 370, 9: 560}
RL_Hybrid = {0: 460, 1: 430, 2: 329, 3: 308, 4: 470, 5: 320, 6: 400, 7: 160, 8: 420, 9: 650}
WORL_RTGS = {0: 253, 1: 252, 2: 134, 3: 103, 4: 255, 5: 145, 6: 222, 7: 125, 8: 225, 9: 391}

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
#plt.savefig("50-svmad-svmm-svm-df-blr.jpg")
plt.show()
