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
DRAS = {0: 235, 1: 220, 2: 85, 3: 105, 4: 140, 5: 135, 6: 215, 7: 125, 8: 190, 9: 350}
mDRL = {0: 220, 1: 215, 2: 140, 3: 110, 4: 186, 5: 155, 6: 240, 7: 136, 8: 265, 9: 390}
MAGRL = {0: 310, 1: 340, 2: 120, 3: 150, 4: 210, 5: 180, 6: 290, 7: 130, 8: 280, 9: 460}
MIGMPS = {0: 350, 1: 300, 2: 170, 3: 120, 4: 330, 5: 220, 6: 260, 7: 200, 8: 300, 9: 500}
RL_Hybrid = {0: 410, 1: 360, 2: 340, 3: 210, 4: 420, 5: 300, 6: 300, 7: 240, 8: 390, 9: 600}

WORL_RTGS = {0: 158, 1: 152, 2: 119, 3: 105, 4: 162, 5: 129, 6: 189, 7: 112, 8: 146, 9: 264}

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
# 设置竖坐标的刻度，最大值为10000，间隔为1000
# plt.yticks(np.arange(0, 8001, 1000))
plt.legend(ncol=3, fontsize=10)
plt.savefig("50-svmad-svmm-svm-df-blr.jpg")
plt.show()
