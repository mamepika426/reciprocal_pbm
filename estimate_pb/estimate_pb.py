import os
import sys
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../') # constsをインポートするために必要
from consts import DATA_DIR, FIG_DIR, NUM_SHEETS, S, POW_SEND, POW_REPLY, MAX_ITER_SEND, MAX_ITER_REPLY, THRESHOLD_SEND, THRESHOLD_REPLY
from estimate_pb.regression_em import regressionEM_send, regressionEM_reply


def estimate_pb(train_logdata: list, sender_profiles: np.ndarray, receiver_profiles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    trainデータから送信時、返信時のポジションバイアスを推定する.
    推定結果と対数尤度のグラフをdata/以下に保存する.
    @param train_logdata: logdataのうち、trainに使う部分
    @param sender_profiles: メッセージ送信先である性別すべてのユーザのprofile [(ユーザ数, 100)]
    @param receiver_profiles: メッセージ送信先となる性別すべてのユーザのprofile [(ユーザ数, 100)]
    @return theta_send_est: 推定された送信時のポジションバイアス [(slate_size)]
    @return theta_reply_est: 推定された返信時のポジションバイアス [(max_sender_position+1)]
    """
    # 送信時のポジションバイアス推定
    theta_send_est, log_likelihood_list_send = regressionEM_send(
                                               train_logdata,
                                               sender_profiles,
                                               receiver_profiles,
                                               S,
                                               MAX_ITER_SEND,
                                               THRESHOLD_SEND)

    # 対数尤度のグラフ
    plt.subplots(1, figsize=(8, 6))
    plt.plot(range(len(log_likelihood_list_send)), log_likelihood_list_send, label="log_likelihood_send", linewidth=3)
    plt.xlabel("num of loops", fontdict=dict(size=20))
    plt.ylabel("log_likelihood_send", fontdict=dict(size=20))
    plt.savefig(os.path.join(FIG_DIR, 'log_likelihood_send.pdf'))

    # 真の送信時のポジションバイアス(左図)と推定された送信時のポジションバイアス(右図)のグラフ
    def pos_to_bias_send(pos: int) -> float:
        return (0.9 / pos) ** POW_SEND

    theta_send_true = np.vectorize(pos_to_bias_send)(np.arange(1, 11))
    positions = np.arange(1, 11)
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].bar(positions, theta_send_true, align="center")
    ax[1].bar(positions, theta_send_est[1:], align="center")
    ax[0].set_ylim(0, 1)
    ax[1].set_ylim(0, 1)
    plt.savefig(os.path.join(FIG_DIR, 'pb_send.pdf'))

    # 返信時のポジションバイアス推定
    theta_reply_est, log_likelihood_list_reply, max_sender_position = regressionEM_reply(
                                                                      train_logdata,
                                                                      sender_profiles,
                                                                      receiver_profiles,
                                                                      MAX_ITER_REPLY,
                                                                      THRESHOLD_REPLY)

    # 対数尤度を表示
    plt.subplots(1, figsize=(8, 6))
    plt.plot(range(len(log_likelihood_list_reply)), log_likelihood_list_reply, label="log_likelihood_reply", linewidth=3)
    plt.xlabel("num of loops", fontdict=dict(size=20))
    plt.ylabel("log_likelihood_reply", fontdict=dict(size=20))
    plt.savefig(os.path.join(FIG_DIR, 'log_likelihood_reply.pdf'))

    # 真の返信時のポジションバイアス(左図)と推定された返信時のポジションバイアス(右図)を表示
    def pos_to_bias_reply(pos: int) -> float:
        return (0.9 / pos) ** POW_REPLY

    theta_reply_true = np.vectorize(pos_to_bias_reply)(np.arange(1, max_sender_position + 1))
    positions = np.arange(1, max_sender_position + 1)
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].bar(positions, theta_reply_true, align="center")
    ax[1].bar(positions, theta_reply_est[1:], align="center")
    ax[0].set_ylim(0, 1)
    ax[1].set_ylim(0, 1)
    plt.savefig(os.path.join(FIG_DIR, 'pb_reply.pdf'))

    # 送信時、返信時のポジションバイアスの推定値を返す
    return theta_send_est, theta_reply_est
