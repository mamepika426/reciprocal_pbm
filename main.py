import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from consts import (DATA_DIR, DATA_DIR_M2F, DATA_DIR_F2M, FIG_DIR, NUM_SHEETS_M2F, NUM_SHEETS_F2M
                    , NUM_TRAIN_SHEETS_M2F, NUM_TRAIN_SHEETS_F2M, S, USE_TRUE_PB, POW_SEND, POW_REPLY, N_EPOCHS)
from train import train_3rankers
from dataset import DatasetBothSide, DatasetOneSide
from estimate_pb.estimate_pb import estimate_pb
from gen_data import gen_a_sheet

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
rel_male2female = np.load(os.path.join(DATA_DIR, 'rel_male2female.npy'))
rel_female2male = np.load(os.path.join(DATA_DIR, 'rel_female2male.npy'))

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

# both, m2f, f2mの3つのデータセットに対してtrain_3rankersを実行
dict_both = train_3rankers(train_both, test_both)
score_fn_naive_both = dict_both['score_fn_naive']
score_fn_ips_both = dict_both['score_fn_ips']
score_fn_ideal_both = dict_both['score_fn_ideal']

# dict_m2f = train_3rankers(train_m2f, test_m2f)
# score_fn_naive_m2f = dict_m2f['score_fn_naive']
# score_fn_ips_m2f = dict_m2f['score_fn_ips']
# score_fn_ideal_m2f = dict_m2f['score_fn_ideal']

# dict_f2m = train_3rankers(train_f2m, test_f2m)
# score_fn_naive_f2m = dict_f2m['score_fn_naive']
# score_fn_ips_f2m = dict_f2m['score_fn_ips']
# score_fn_ideal_f2m = dict_f2m['score_fn_ideal']

# 実験1: naive vs ips vs ideal(bothのみ)
plt.subplots(1, figsize=(8, 6))
plt.plot(range(N_EPOCHS), dict_both['ndcg_score_list_naive'], label="Naive", linewidth=3, linestyle="dashed")
plt.plot(range(N_EPOCHS), dict_both['ndcg_score_list_ips'], label="IPS", linewidth=3, linestyle="dashdot")
plt.plot(range(N_EPOCHS), dict_both['ndcg_score_list_ideal'], label="Ideal", linewidth=3)

plt.title("Test nDCG@10 Curve With Different Estimators (pow_true=({}, {}))".format(POW_SEND, POW_REPLY), fontdict=dict(size=15))
plt.xlabel("Number of Epochs", fontdict=dict(size=20))
plt.ylabel("Test nDCG@10", fontdict=dict(size=20))
plt.tight_layout()
plt.legend(loc="best", fontsize=20)
plt.savefig(os.path.join(FIG_DIR, "ndcg_both.pdf"))

# 実験2: both vs oneside
# 各ユーザについての、まだメッセージを送っていない相手のリスト
candidates = []
for sender in range(len(male_profiles)):
    candidates.append(np.arange(len(female_profiles)))

for n in range(NUM_TRAIN_SHEETS_M2F):
    for item in logdata_m2f[n].itertuples():
        if item.送信有無 == 1:
            candidates[item.送信者] = np.setdiff1d(candidates[item.送信者], item.受信者)

# テストシート上の各送信者に対して、それぞれのスコアリング関数通りに表示するといくつのペアができるか
sum_naive_both = 0
sum_ips_both = 0
sum_ideal_both = 0
sum_naive_oneside = 0
sum_ips_oneside = 0
sum_ideal_oneside = 0

for n in range(NUM_SHEETS_M2F - NUM_TRAIN_SHEETS_M2F):
    sheet = logdata_m2f[NUM_TRAIN_SHEETS_M2F + n]
    sender = sheet['送信者'][0]

    # score_fn_naive_bothの性能評価
    # np.tileで同じ配列を縦に繰り返し: https://note.nkmk.me/python-numpy-tile/
    sender_repeat = np.tile(male_profiles[sender], (len(candidates[sender]), 1))
    feature_a_sender = np.hstack((sender_repeat, female_profiles[candidates[sender]]))
    score_list = score_fn_naive_both(torch.FloatTensor(feature_a_sender)).tolist()
    candidates_with_scores = list(zip(candidates[sender], score_list))
    output = sorted(candidates_with_scores, key=lambda x:x[1], reverse=True)
    receivers = [i[0] for i in output[:10]]
    # スコアリング関数通りに表示した場合のクリックログ
    sheet, _ = gen_a_sheet(sender, np.array(receivers), [[] for female in range(len(female_profiles))], rel_male2female, rel_female2male)
    sum_naive_both += sum(sheet['返信有無'])

    # score_fn_ips_bothの性能評価
    score_list = score_fn_ips_both(torch.FloatTensor(feature_a_sender)).tolist()
    candidates_with_scores = list(zip(candidates[sender], score_list))
    output = sorted(candidates_with_scores, key=lambda x:x[1], reverse=True)
    receivers = [i[0] for i in output[:10]]
    # スコアリング関数通りに表示した場合のクリックログ
    sheet, _ = gen_a_sheet(sender, np.array(receivers), [[] for female in range(len(female_profiles))], rel_male2female, rel_female2male)
    sum_ips_both += sum(sheet['返信有無'])

    # score_fn_idealの性能評価
    score_list = score_fn_ideal_both(torch.FloatTensor(feature_a_sender)).tolist()
    candidates_with_scores = list(zip(candidates[sender], score_list))
    output = sorted(candidates_with_scores, key=lambda x:x[1], reverse=True)
    receivers = [i[0] for i in output[:10]]
    # スコアリング関数通りに表示した場合のクリックログ
    sheet, _ = gen_a_sheet(sender, np.array(receivers), [[] for female in range(len(female_profiles))], rel_male2female, rel_female2male)
    sum_ideal_both += sum(sheet['返信有無'])

print("naive_both: ", sum_naive_both)
print("ips_both: ", sum_ips_both)
print("ideal_both: ", sum_ideal_both)
