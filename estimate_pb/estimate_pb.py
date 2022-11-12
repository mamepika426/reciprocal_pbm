import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../') # constsをインポートするために必要
from consts import DATA_DIR, FIG_DIR, NUM_SHEETS, S, POW_SEND, POW_REPLY
from regression_em import regressionEM_send, regressionEM_reply

# 生成されたデータをロード
logdata = []
for n in range(NUM_SHEETS):
    csv_file_path = os.path.join(DATA_DIR, 'sheet{0}.csv'.format(n))
    df = pd.read_csv(csv_file_path, header=0)
    logdata.append(df)

male_profiles = np.load(os.path.join(DATA_DIR, 'male_profiles.npy'))
female_profiles = np.load(os.path.join(DATA_DIR, 'female_profiles.npy'))


# 送信時のポジションバイアス推定
theta_est, log_likelihood_list = regressionEM_send(logdata, male_profiles, female_profiles, S, max_iter=10, threshold=0.8)

# 対数尤度を表示
plt.subplots(1, figsize=(8, 6))
plt.plot(range(len(log_likelihood_list)), log_likelihood_list, label="log_likelihood_list", linewidth=3)
plt.xlabel("num of loops", fontdict=dict(size=20))
plt.ylabel("log likelihood", fontdict=dict(size=20))
plt.savefig(os.path.join(FIG_DIR, 'log_likelihood_send.pdf'))

# 真の送信時のポジションバイアス(左図)と推定された送信時のポジションバイアス(右図)を表示
def pos_to_bias_send(pos: int) -> float:
    return (0.9 / pos) ** POW_SEND

theta_true = np.vectorize(pos_to_bias_send)(np.arange(1, 11))
positions = np.arange(1, 11)
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].bar(positions, theta_true, align="center")
ax[1].bar(positions, theta_est[1:], align="center")
ax[0].set_ylim(0, 1)
ax[1].set_ylim(0, 1)
plt.savefig(os.path.join(FIG_DIR, 'pb_send.pdf'))

# 返信時のポジションバイアス推定
theta_est_reply, log_likelihood_list_reply, max_sender_position = regressionEM_reply(logdata, male_profiles, female_profiles, max_iter=15, threshold=0.2)

# 対数尤度を表示
plt.subplots(1, figsize=(8, 6))
plt.plot(range(len(log_likelihood_list_reply)), log_likelihood_list_reply, label="log_likelihood_list_reply", linewidth=3)
plt.xlabel("num of loops", fontdict=dict(size=20))
plt.ylabel("log likelihood", fontdict=dict(size=20))
plt.savefig(os.path.join(FIG_DIR, 'log_likelihood_reply.pdf'))

# 真の返信時のポジションバイアス(左図)と推定された返信時のポジションバイアス(右図)を表示
def pos_to_bias_reply(pos: int) -> float:
    return (0.9 / pos) ** POW_REPLY

theta_true = np.vectorize(pos_to_bias_reply)(np.arange(1, max_sender_position + 1))
positions = np.arange(1, max_sender_position + 1)
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].bar(positions, theta_true, align="center")
ax[1].bar(positions, theta_est_reply[1:], align="center")
ax[0].set_ylim(0, 1)
ax[1].set_ylim(0, 1)
plt.savefig(os.path.join(FIG_DIR, 'pb_reply.pdf'))
