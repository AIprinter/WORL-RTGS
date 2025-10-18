import matplotlib.pyplot as plt

# 数据 Alibaba Data 50 GPU
tasks = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2200,  2400, 2600, 2800, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

pcc_busy = [0.4147, 0.4487, 0.3732, 0.5243, 0.4233, 0.4536, 0.4931, 0.4638, 0.3632, 0.4074, 0.3985, 0.5329, 0.4949, 0.5058,0.4632, 0.4118, 0.3840, 0.4218, 0.4155, 0.4306, 0.4173, 0.4834, 0.3482, 0.5006, 0.3672, 0.4620, 0.3210]

pcc_idle = [0.4192, 0.1627, 0.3724, 0.3622, 0.4153, 0.3532, 0.4133, 0.4690, 0.3990, 0.4913, 0.4029, 0.3937, 0.4164, 0.4899, 0.4145, 0.5158, 0.5094, 0.4673, 0.3967, 0.4905, 0.4104, 0.4794, 0.3849, 0.5374, 0.4412, 0.3756, 0.3205]

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
