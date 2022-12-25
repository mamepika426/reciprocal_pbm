import sys
from typing import Optional

import numpy as np
import torch

from torch import LongTensor, nn
from torch.utils.data import DataLoader
from pytorchltr.evaluation.dcg import ndcg

from consts import S
from dataset import BaseDataset
from gen_data import gen_a_sheet

def evaluate_test_performance(score_fn: nn.Module, test: BaseDataset, S: int, test_batch_size: int) -> float:
    """
    与えられたmodelのランキング性能をテストデータにおける真の嗜好度合情報をつかってnDCG@10で評価する.
    """
    loader = DataLoader(test, test_batch_size, shuffle=False, drop_last=True)
    ndcg_score = 0.0
    n = LongTensor(np.ones(test_batch_size)* (S-1))
    for batch in loader:
        true_score = batch['true_score']
        ndcg_score += ndcg(
            score_fn(batch['features']), true_score, n, k=10, exp=False
        ).sum()

    return float(ndcg_score / len(test))


def online_performance_per_sender(sender: int, sender_profiles: np.ndarray, receiver_profiles: np.ndarray,
                                  rel_sender2receiver: np.ndarray, rel_receiver2sender: np.ndarray, candidates: np.ndarray, aggregation: str,
                                  score_fn_sender2receiver: nn.Module, score_fn_receiver2sender: Optional[nn.Module] = None,) -> int:
    """
    送信者senderに対してスコアリング関数通りに受信者を表示した時に成立するマッチ数を返す
    @param sender: 送信者
    @param sender_profiles: メッセージ送信源となる性別すべてのユーザのprofile [(ユーザ数, 100)]
    @param receiver_profiles: メッセージ送信先となる性別すべてのユーザのprofile [(ユーザ数, 100)]
    @param rel_sender2receiver: メッセージ送信源の性別から送信先の性別への嗜好度合いが格納された配列
    @param rel_receiver2sender: メッセージ送信先の性別から送信源の性別への嗜好度合いが格納された配列
    @param candidates: 各ユーザについての、まだメッセージを送っていない相手のリスト
    @param aggregation: 'both', 'product', 'harmonic'のいずれかを指定
            both: 返信があったペアのみを学習に使ったスコアリング関数
            product: 送信ログのみを学習に使ったスコアリング関数同士を積で結合
            harmonic: 送信ログのみを学習に使ったスコアリング関数同士を調和平均で結合
    @param score_fn_sender2receiver: 送信者から受信者へのスコアリング関数
    @param score_fn_receiver2sender: 送信者から受信者へのスコアリング関数
    """
    assert aggregation in [
        "both",
        "product",
        "harmonic",
    ], f"aggregation must be 'both', 'product', or 'harmonic', but {aggregation} is given"

    sender_repeat = np.tile(sender_profiles[sender], (len(candidates[sender]), 1))
    feature_sender2receiver = np.hstack((sender_repeat, receiver_profiles[candidates[sender]]))
    if aggregation == "both":
        score_list = score_fn_sender2receiver(torch.FloatTensor(feature_sender2receiver)).tolist()
    else:
        if score_fn_receiver2sender == None:
            sys.exit("片側だけのスコアリング関数同士を結合する場合は、2つのスコアリング関数を用意してください.")
        feature_receiver2sender = np.hstack((receiver_profiles[candidates[sender]], sender_repeat))
        score_sender2receiver = score_fn_sender2receiver(torch.FloatTensor(feature_sender2receiver)).detach().numpy()
        score_receiver2sender = score_fn_receiver2sender(torch.FloatTensor(feature_receiver2sender)).detach().numpy()
        if aggregation == "product":
            score_list = (score_sender2receiver * score_receiver2sender).tolist()
        elif aggregation == "harmonic":
            score_sum = score_sender2receiver + score_receiver2sender
            score_product = score_sender2receiver * score_receiver2sender
            score_list = ((2*score_product) / score_sum).tolist()

    candidates_with_scores = list(zip(candidates[sender], score_list))
    output = sorted(candidates_with_scores, key=lambda x:x[1], reverse=True)
    receivers = [i[0] for i in output[:10]]
    # スコアリング関数通りに表示した場合のクリックログ
    sheet, _ = gen_a_sheet(sender,
                           np.array(receivers),
                           [[] for receiver in range(len(receiver_profiles))],
                           rel_sender2receiver,
                           rel_receiver2sender)

    return sum(sheet['返信有無'])
