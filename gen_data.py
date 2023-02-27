# ユーザ、メッセージ送信ログについてのデータを生成する関数群
import os
import random

import numpy as np
import pandas as pd

from typing import List, Tuple
from scipy.stats import bernoulli

from consts import (DATA_DIR, DATA_DIR_M2F, DATA_DIR_F2M, DATA_DIR,FIG_DIR, NUM_MALES, NUM_FEMALES, NUM_CLUSTERS
                   , S, NUM_SHEETS_M2F, NUM_SHEETS_F2M, POW_SEND, POW_REPLY)


def gen_users(num_users: int, num_clusters: int) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    今考えている性別(性別Aとする)の全ユーザのprofile, preferenceを表す100次元ベクトルを混合正規分布から生成する関数
    @param num_users: 性別Aのユーザ数
    @param num_clusters: ユーザ生成に使うクラスター数
    @return profiles: 性別Aの各ユーザのprofileが格納された配列 [(num_users, 100)]
    @return preferences: 性別Aの各ユーザのpreferenceが格納された配列 [(num_users, 100)]
    @return profile_cluster_list: 性別Aの各ユーザのprofileが属するクラスターが格納された配列 [(num_users,)]
    """
    clusters_list = range(num_clusters)
    cluster_features_list = [random.choices([0, 1, 2], k=100, weights=[1/3, 1/3, 1/3]) for cluster in range(num_clusters)]
    cluster_choice_proba = np.ones(num_clusters) / num_clusters
    cluster_list = random.choices(clusters_list, k=num_users, weights=cluster_choice_proba)
    
    # preference_cluster_list = random.choices(clusters_list, k=num_users, weights=[0.5, 0.5])
    
    profiles = np.empty((num_users, 100))
    preferences = np.empty((num_users, 100))
    for user in range(num_users):
        profiles[user] = np.random.multivariate_normal(cluster_features_list[cluster_list[user]], np.identity(100))
        preferences[user] = np.random.multivariate_normal(cluster_features_list[cluster_list[user]], np.identity(100))

    return profiles, preferences, cluster_list


def gen_relevances(profiles: np.ndarray, preferences: np.ndarray) -> np.ndarray:
    """
    嗜好度合いを作成する関数
    @param profiles: メッセージ送信先となる性別すべてのユーザのprofile [(ユーザ数, 100)]
    @param preferences: メッセージ送信源となる性別すべてのユーザのpreference [(ユーザ数, 100)]
    @return 嗜好度合いが格納された配列
    """
    # preferenceとprofileのユークリッド距離の逆数を計算
    inv_distances_dict = {}
    for i in range(len(preferences)):
        for j in range(len(profiles)):
            inv_distances_dict.update({i*len(profiles) + j: 1 / (np.linalg.norm(preferences[i] - profiles[j], ord = 2))})

    # beta(0.7, 0.7)に従うように圧縮
    beta = np.random.beta(0.7, 0.7, len(preferences)*len(profiles))
    beta_sorted = np.sort(beta)

    inv_distances_dict_sorted = sorted(inv_distances_dict.items(), key=lambda x:x[1]) # <- [(index, inv_distance)]
    relevances_dict = {}
    for i, (index, _) in enumerate(inv_distances_dict_sorted):
        relevances_dict.update( {index: beta_sorted[i]} )
    
    relevances_dict_sorted = sorted(relevances_dict.items(), key=lambda x:x[0])
    relevances = [relevance for _, relevance in relevances_dict_sorted]
    
    return np.array(relevances).reshape(len(preferences), len(profiles))


def gen_a_sheet(sender: int, receivers: np.ndarray, messaged_by: list
                , rel_sender2receiver: np.ndarray, rel_receiver2sender: np.ndarray) -> Tuple[pd.DataFrame, list]:
    """
    1検索分のシートを作成する関数
    @param sender: このシートの検索者
    @param receivers: 今回の検索で表示されたユーザーの配列
    @param messaged_by: 各受信者が、すでにメッセージを受け取った相手を格納している配列 [(メッセージ送信先の性別のユーザ数, *)]
    @param rel_sender2receiver: メッセージ送信源の性別から送信先の性別への嗜好度合いが格納された配列
    @param rel_receiver2sender: メッセージ送信先の性別から送信源の性別への嗜好度合いが格納された配列
    @return sheet: 1検索についての情報をdfにしたもの
    @return messaged_by: messaged_byに今回の検索結果を反映したもの [(メッセージ送信先の性別のユーザ数, *)]
    """
    # 送信者, 受信者カラム
    sender_values = [sender] * len(receivers)
    receiver_values = receivers.tolist()

    # 送信者, 受信者位置カラム
    receiver_positions = range(1, len(receivers) + 1)
    sender_positions = [len(messaged_by[receiver])+1 for receiver in np.nditer(receivers)]
    
    # 嗜好度合いカラム
    rel_sender2receiver_values = [rel_sender2receiver[sender][receiver] for receiver in receivers]
    rel_receiver2sender_values = [rel_receiver2sender[receiver][sender] for receiver in receivers]

    # シート作成
    sheet_dict = {'送信者': sender_values, 
                  '受信者': receiver_values,
                  '受信者位置': receiver_positions,
                  '送信者位置': sender_positions,
                  'gamma_sender': rel_sender2receiver_values,
                  'gamma_receiver': rel_receiver2sender_values}
    sheet = pd.DataFrame(sheet_dict)

    # 送信有無カラム
    sheet['送信有無'] = bernoulli.rvs( sheet['gamma_sender'] * ((0.9/sheet['受信者位置']) ** POW_SEND) )
    sheet['返信有無'] = bernoulli.rvs( sheet['gamma_receiver'] * ((0.9/sheet['送信者位置']) ** POW_REPLY) ) * sheet['送信有無']

    # 完成したシートを確認して、メッセージ送信があればmessaged_byを更新しておく
    for item in sheet[sheet['送信有無']==1].itertuples():
        messaged_by[item.受信者].append(item.送信者)
        
    return sheet, messaged_by


def gen_logdata(profiles: np.ndarray, preferences: np.ndarray, S: int, num_search: int
                , rel_sender2receiver: np.ndarray, rel_receiver2sender: np.ndarray,) -> list:
    """
    ログデータを作成する関数
    @param profiles: メッセージ送信先となる性別すべてのユーザのprofile [(ユーザ数, 100)]
    @param preferences: メッセージ送信源となる性別すべてのユーザのpreference [(ユーザ数, 100)]
    @param S: slate size
    @param num_search: 検索総回数
    @param rel_sender2receiver: メッセージ送信源の性別から送信先の性別への嗜好度合いが格納された配列
    @param rel_receiver2sender: メッセージ送信先の性別から送信源の性別への嗜好度合いが格納された配列
    @return logdata: 1検索についての情報をdfにしたものをnum_search枚もつリスト
    """
    # 送信者ごとに、まだメッセージを送っていない相手を格納するリスト
    candidates = []
    for sender in range(len(preferences)):
        candidates.append(np.arange(len(profiles)))
    
    # 受信者ごとに送信源のユーザを格納するリスト
    messaged_by = [[] for i in range(len(profiles))]

    # 各検索での送信者、受信者を保存しておく
    senders_list = []
    receivers_list = []
    for n in range(num_search):
        # 送信者を1人ランダムに選択
        sender = np.random.randint(len(preferences))
        # S人の受信者をランダムに選択し、検索済みの相手は候補から外しておく
        receivers = np.random.choice(candidates[sender], S, replace=False)
        candidates[sender] = np.setdiff1d(candidates[sender], receivers)

        senders_list.append(sender)
        receivers_list.append(receivers)
    
    # ログデータ生成
    logdata = []
    for n in range(num_search):
        sheet, messaged_by = gen_a_sheet(senders_list[n], receivers_list[n], messaged_by
                                         , rel_sender2receiver, rel_receiver2sender)
        logdata.append(sheet)

    return logdata


if __name__ == "__main__":
    # data, figディレクトリが存在しなければ作成
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    if not os.path.exists(DATA_DIR_M2F):
        os.mkdir(DATA_DIR_M2F)

    if not os.path.exists(DATA_DIR_F2M):
        os.mkdir(DATA_DIR_F2M)

    if not os.path.exists(FIG_DIR):
        os.mkdir(FIG_DIR)

    # profile, preference, 嗜好度合い, ログデータ作成
    male_profiles, male_preferences, male_cluster_list = gen_users(NUM_MALES, NUM_CLUSTERS)
    female_profiles, female_preferences, female_cluster_list = gen_users(NUM_FEMALES, NUM_CLUSTERS)
    rel_male2female = gen_relevances(female_profiles, male_preferences)
    rel_female2male = gen_relevances(male_profiles, female_preferences)

    logdata_m2f = gen_logdata(female_profiles, male_preferences, S, NUM_SHEETS_M2F, rel_male2female, rel_female2male)
    logdata_f2m = gen_logdata(male_profiles, female_preferences, S, NUM_SHEETS_F2M, rel_female2male, rel_male2female)

    # データ保存
    for n in range(len(logdata_m2f)):
        csv_file_path = os.path.join(DATA_DIR_M2F, 'sheet{0}.csv'.format(n))
        logdata_m2f[n].to_csv(csv_file_path, index=False)

    for n in range(len(logdata_f2m)):
        csv_file_path = os.path.join(DATA_DIR_F2M, 'sheet{0}.csv'.format(n))
        logdata_f2m[n].to_csv(csv_file_path, index=False)

    np.save(os.path.join(DATA_DIR, 'male_profiles'), male_profiles)
    np.save(os.path.join(DATA_DIR, 'female_profiles'), female_profiles)
    np.save(os.path.join(DATA_DIR, 'rel_male2female'), rel_male2female)
    np.save(os.path.join(DATA_DIR, 'rel_female2male'), rel_female2male)
