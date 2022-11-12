import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.optim import Adam

from consts import (DATA_DIR, FIG_DIR, NUM_SHEETS, NUM_TRAIN_SHEETS, S,
                    TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, HIDDEN_LAYER_SIZES, LEARNING_RATE, N_EPOCHS)
from model import MLPScoreFunc
from train import train_ranker
from dataset import MyDataset

# 生成されたデータをロード
logdata = []
for n in range(NUM_SHEETS):
    csv_file_path = os.path.join(DATA_DIR, 'sheet{0}.csv'.format(n))
    df = pd.read_csv(csv_file_path, header=0)
    logdata.append(df)

male_profiles = np.load(os.path.join(DATA_DIR, 'male_profiles.npy'))
female_profiles = np.load(os.path.join(DATA_DIR, 'female_profiles.npy'))

# 後で絶対シャッフル
train = MyDataset(logdata[:NUM_TRAIN_SHEETS], male_profiles, female_profiles, S)
test = MyDataset(logdata[NUM_TRAIN_SHEETS:], male_profiles, female_profiles, S)


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
    train=train,
    test=test,
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
    train=train,
    test=test,
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
    train=train,
    test=test,
    train_batch_size=TRAIN_BATCH_SIZE,
    test_batch_size=TEST_BATCH_SIZE,
    n_epochs=N_EPOCHS,
)

plt.subplots(1, figsize=(8, 6))
plt.plot(range(N_EPOCHS), ndcg_score_list_naive, label="Naive", linewidth=3, linestyle="dashed")
plt.plot(range(N_EPOCHS), ndcg_score_list_ips, label="IPS", linewidth=3, linestyle="dashdot")
plt.plot(range(N_EPOCHS), ndcg_score_list_ideal, label="Ideal", linewidth=3)

plt.title("Test nDCG@10 Curve With Different Estimators (pow_true=1)", fontdict=dict(size=15))
plt.xlabel("Number of Epochs", fontdict=dict(size=20))
plt.ylabel("Test nDCG@10", fontdict=dict(size=20))
plt.tight_layout()
plt.legend(loc="best", fontsize=20)
plt.savefig(os.path.join(FIG_DIR, "ndcg.pdf"))
