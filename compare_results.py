import os
import numpy as np
import matplotlib.pyplot as plt

# this script generates a single graph of all results available in the ./data folder

exp_runs = [f.path for f in os.scandir("data") if f.is_dir() ]

print(exp_runs)

for exp in exp_runs:

    exp_avg_perf = os.path.join(exp, "avg_perf.npy")
    print(exp_avg_perf)

    if os.path.isfile(exp_avg_perf):
        exp_avg_perf_data = np.load(exp_avg_perf)

        plt.plot(exp_avg_perf_data)

plt.legend(['Uniform sampling', '1/19 Option-action', 'Only actions'])
plt.show()
