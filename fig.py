import argparse
from typing import List
import matplotlib
from matplotlib import legend
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes

parser = argparse.ArgumentParser()
parser.add_argument("server_log")
args = parser.parse_args()
print(args)

RATE = []
TIMES = []
RUNNING_BS = []
STAGED_BS = []
TARGET_STAGED_BS = []
EVICTED = []
DECODE_LEN_EMA = []
DECODE_AVG = []

N_STARTED = []
N_FINISHED = []
with open(args.server_log) as f:
    _offset = -9999
    for l in f.readlines():
        if 'AdaptiveBatchsizeRouter usage' in l:
            RATE.append(float(l.split()[2][:-1]) / 100)
            _offset = 0
            if len(EVICTED) < len(RATE):
                EVICTED.append(0)
            if len(DECODE_LEN_EMA) < len(RATE):
                DECODE_LEN_EMA.append(DECODE_LEN_EMA[-1] if DECODE_LEN_EMA else 0)
            if len(DECODE_AVG) < len(RATE):
                DECODE_AVG.append(DECODE_AVG[-1] if DECODE_AVG else 0)
        else:
            _offset += 1

        if 'AdaptiveBatchsizeRouter(target_staged_bs' in l:
            STAGED_BS.append(int(l.split()[2].split('=')[1][:-1]))
            RUNNING_BS.append(int(l.split()[3].split('=')[1][:-1]))
            TARGET_STAGED_BS.append(int(l.split()[0].split('=')[1][:-1]))

        if _offset == 2:    # timestamp
            try:
                TIMES.append(float(l.strip()))
            except ValueError:
                _offset -= 1

        if 'will evict' in l:
            EVICTED.append(int(l.split()[2]))

        if 'decoded len average' in l:
            DECODE_AVG.append(float(l.split()[3]))

        if 'history_decode_len_ema' in l:
            DECODE_LEN_EMA.append(float(l.split()[1]))

        if 'n_finished' in l:
            N_FINISHED.append(float(l.split()[1]))
            N_STARTED.append(float(l.split()[4]))




for i in range(1, len(TIMES)):
    TIMES[i] -= TIMES[0]
TIMES[0] = 0

STEPS = list(range(len(RATE)))

fig, axs = plt.subplots(5, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [2, 2, 2, 0.25, 0.25]})
axs: List[matplotlib.axes.Axes] = axs
axs[0].set_title('vram usage rate')
axs[0].plot(TIMES, RATE, linewidth=1)

axs[1].set_title('bs')
axs[1].plot(TIMES, RUNNING_BS, linewidth=1)
axs[1].plot(TIMES, STAGED_BS, linewidth=1)
axs[1].plot(TIMES, TARGET_STAGED_BS, linewidth=1)
ax1_1 = axs[1].twinx()
ax1_1.plot(TIMES, EVICTED, linewidth=0.5, color='red')
# ax1_2 = axs[1].twinx()
# ax1_2.plot(TIMES, np.cumsum(EVICTED), linewidth=0.5, color='gray')

# TIMES.append(0)
axs[2].set_title('len')
axs[2].plot(TIMES, DECODE_LEN_EMA, linewidth=1)
axs[2].plot(TIMES, DECODE_AVG, linewidth=1)

# cmap = plt.get_cmap('hot')
axs[3].set_title('finish')
axs[3].bar(TIMES, np.array(N_FINISHED[:-1]) > 0, color=['blue' if x == 1 else 'orange' if x < 5 else 'red' for x in N_FINISHED[:-1]])
# axs[3].bar(TIMES, [1] * len(TIMES), color=[cmap(np.log(x)) for x in N_FINISHED[:-1]])
axs[4].set_title('start')
axs[4].bar(TIMES, np.array(N_STARTED[:-1]) > 0, color=['blue' if x == 1 else 'orange' if x < 5 else 'red' for x in N_STARTED[:-1]])
# axs[4].bar(TIMES, [1] * len(TIMES), color=[cmap(np.log(x)) for x in N_STARTED[:-1]])



plt.subplots_adjust(hspace=0.35, top=0.95, bottom=0.05, left=0.1, right=0.9)
plt.savefig("fig.png", dpi=300)
