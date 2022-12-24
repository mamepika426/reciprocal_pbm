import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.optim import Adam

from consts import (DATA_DIR, DATA_DIR_M2F, DATA_DIR_F2M, FIG_DIR, NUM_SHEETS_M2F, NUM_SHEETS_F2M
                    , NUM_TRAIN_SHEETS_M2F, NUM_TRAIN_SHEETS_F2M, S, USE_TRUE_PB, POW_SEND, POW_REPLY
                    , TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, HIDDEN_LAYER_SIZES, LEARNING_RATE, N_EPOCHS)
from model import MLPScoreFunc
from train import train_ranker
from dataset import DatasetBothSide, DatasetOneSide
from estimate_pb.estimate_pb import estimate_pb

# 生成されたデータをロード
logdata_m2f = []
logdata_f2m = []
for n in range(NUM_SHEETS_M2F):
    csv_file_path = os.path.join(DATA_DIR_M2F, 'sheet{0}.csv'.format(n))
    df = pd.read_csv(csv_file_path, header=0)
    logdata_m2f.append(df)

for n in range(NUM_SHEETS_F2M):
    csv_file_path = os.path.join(DATA_DIR_F2M, 'sheet{0}.csv'.format(n))
    df = pd.read_csv(csv_file_path, header=0)
    logdata_f2m.append(df)

male_profiles = np.load(os.path.join(DATA_DIR, 'male_profiles.npy'))
female_profiles = np.load(os.path.join(DATA_DIR, 'female_profiles.npy'))

# 真のポジションバイアスが使える場合
if USE_TRUE_PB:
    # 後で絶対シャッフル
    train_both = DatasetBothSide(logdata_m2f[:NUM_TRAIN_SHEETS_M2F], male_profiles, female_profiles, S)
    test_both = DatasetBothSide(logdata_m2f[NUM_TRAIN_SHEETS_M2F:], male_profiles, female_profiles, S)
    train_m2f = DatasetOneSide(logdata_m2f[:NUM_TRAIN_SHEETS_M2F], male_profiles, female_profiles, S)
    test_m2f = DatasetOneSide(logdata_m2f[NUM_TRAIN_SHEETS_M2F:], male_profiles, female_profiles, S)
    train_f2m = DatasetOneSide(logdata_f2m[:NUM_TRAIN_SHEETS_F2M], female_profiles, male_profiles, S)
    test_f2m = DatasetOneSide(logdata_f2m[NUM_TRAIN_SHEETS_F2M:], female_profiles, male_profiles, S)

# 真のポジションバイアスが使えない場合はregression_emにより推定(渡すデータはtrainと同じもの)
if not USE_TRUE_PB:
    theta_send_est, theta_reply_est = estimate_pb(logdata_m2f[:NUM_TRAIN_SHEETS], male_profiles, female_profiles)
    train_both = DatasetBothSide(logdata_m2f[:NUM_TRAIN_SHEETS_M2F], male_profiles, female_profiles, S, theta_send_est, theta_reply_est)
    test_both = DatasetBothSide(logdata_m2f[NUM_TRAIN_SHEETS_M2F:], male_profiles, female_profiles, S)
    train_m2f = DatasetOneSide(logdata_m2f[:NUM_TRAIN_SHEETS_M2F], male_profiles, female_profiles, S, theta_send_est, theta_reply_est)
    test_m2f = DatasetOneSide(logdata_m2f[NUM_TRAIN_SHEETS_M2F:], male_profiles, female_profiles, S, theta_send_est, theta_reply_est)
    train_f2m = DatasetOneSide(logdata_f2m[:NUM_TRAIN_SHEETS_F2M], female_profiles, male_profiles, S, theta_send_est, theta_reply_est)
    test_f2m = DatasetOneSide(logdata_f2m[NUM_TRAIN_SHEETS_F2M:], female_profiles, male_profiles, S, theta_send_est, theta_reply_est)

torch.manual_seed(12345)
score_fn = MLPScoreFunc(
    input_size=200,
    hidden_layer_sizes=HIDDEN_LAYER_SIZES,
)
optimizer = Adam(score_fn.parameters(), lr=LEARNING_RATE)

ndcg_score_list_naive = train_ranker(
    score_fn=score_fn,
    optimizer=optimizer,
    estimator="naive",
    train=train_both,
    test=test_both,
    train_batch_size=TRAIN_BATCH_SIZE,
    test_batch_size=TEST_BATCH_SIZE,
    n_epochs=N_EPOCHS,
)

score_fn = MLPScoreFunc(
    input_size=200,
    hidden_layer_sizes=HIDDEN_LAYER_SIZES,
)
optimizer = Adam(score_fn.parameters(), lr=LEARNING_RATE)

ndcg_score_list_ips = train_ranker(
    score_fn=score_fn,
    optimizer=optimizer,
    estimator="ips",
    train=train_both,
    test=test_both,
    train_batch_size=TRAIN_BATCH_SIZE,
    test_batch_size=TEST_BATCH_SIZE,
    n_epochs=N_EPOCHS,
)

torch.manual_seed(12345)
score_fn = MLPScoreFunc(
    input_size=200,
    hidden_layer_sizes=HIDDEN_LAYER_SIZES,
)
optimizer = Adam(score_fn.parameters(), lr=LEARNING_RATE)

ndcg_score_list_ideal = train_ranker(
    score_fn=score_fn,
    optimizer=optimizer,
    estimator="ideal",
    train=train_both,
    test=test_both,
    train_batch_size=TRAIN_BATCH_SIZE,
    test_batch_size=TEST_BATCH_SIZE,
    n_epochs=N_EPOCHS,
)

plt.subplots(1, figsize=(8, 6))
plt.plot(range(N_EPOCHS), ndcg_score_list_naive, label="Naive", linewidth=3, linestyle="dashed")
plt.plot(range(N_EPOCHS), ndcg_score_list_ips, label="IPS", linewidth=3, linestyle="dashdot")
plt.plot(range(N_EPOCHS), ndcg_score_list_ideal, label="Ideal", linewidth=3)

plt.title("Test nDCG@10 Curve With Different Estimators (pow_true=({}, {}))".format(POW_SEND, POW_REPLY), fontdict=dict(size=15))
plt.xlabel("Number of Epochs", fontdict=dict(size=20))
plt.ylabel("Test nDCG@10", fontdict=dict(size=20))
plt.tight_layout()
plt.legend(loc="best", fontsize=20)
plt.savefig(os.path.join(FIG_DIR, "ndcg_both.pdf"))
