import matplotlib.pyplot as plt

# 数据 Alibaba Data 500 GPU
tasks = [   100,     200,  300,    400,   500,   600,   700,   800,   900,   1000, 1200,   1400, 1600, 1800,   2000,  2200,  2400,  2600,  2800,  3000,  4000,  5000,  6000,  7000, 8000, 9000, 10000]

pcc_busy = [0.5008,0.5185,0.5013,0.4564,0.5091,0.4309,0.5085,0.4226,0.4258,0.3668,0.3873,0.4072,0.4348,0.4859,0.4691,0.3990,0.4364,0.4841,0.4113,0.4019,0.4567,0.4122,0.4337,0.4332,0.4645,0.4768,0.4582]

pcc_idle = [0.4906,0.4039,0.5016,0.4270,0.4508,0.4005,0.5210,0.3922,0.4121,0.5378,0.4426,0.5028,0.4746,0.4084,0.5107,0.4927,0.5018,0.5325,0.4490,0.4261,0.5000,0.4934,0.4711,0.4555,0.4322,0.4866,0.4947]

# 画图
plt.figure(figsize=(8,5))
plt.plot(tasks, pcc_busy, marker='o', label="PCC in busy time")
plt.plot(tasks, pcc_idle, marker='s', label="PCC in idle time")

plt.xlabel("Number of tasks")
plt.ylabel("Pearson Correlation Coefficient (PCC)")
#plt.title("Positive Correlation between FTG and SPD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
