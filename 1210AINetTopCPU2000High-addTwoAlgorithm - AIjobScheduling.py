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
DRAS = {9: 350, 10: 258, 11: 311, 12: 299, 13: 307, 14: 504, 15: 270, 16: 607, 17: 825, 18: 849}
mDRL = {9: 390, 10: 305, 11: 392, 12: 315, 13: 412, 14: 575, 15: 335, 16: 667, 17: 705, 18: 905}
MAGRL = {9: 460, 10: 415, 11: 455, 12: 475, 13: 455, 14: 705, 15: 525, 16: 765, 17: 765, 18: 1005}
RL_Hybrid = {9: 600, 10: 405, 11: 565, 12: 525, 13: 605, 14: 935, 15: 695, 16: 865, 17: 841, 18: 1175}
MIGMPS = {9: 500, 10: 435, 11: 465, 12: 515, 13: 490, 14: 705, 15: 440, 16: 895, 17: 795, 18: 1035}

WORL_RTGS = {9: 264, 10: 210, 11: 290, 12: 289, 13: 290, 14: 398, 15: 242, 16: 458, 17: 459, 18: 843}

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

# 设置竖坐标的刻度，最大值为10000，间隔为1000
# plt.yticks(np.arange(0, 22501, 2500))

plt.legend(ncol=3, fontsize=10)
plt.savefig("50-svmad-svm`m-svm-df-blr.jpg")
plt.show()
