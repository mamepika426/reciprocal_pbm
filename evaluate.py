import sys
import random
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
            score_fn(batch['features']), true_score, n, k=S, exp=False
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
    receivers = [i[0] for i in output[:S]]
    # スコアリング関数通りに表示した場合のクリックログ
    sheet, _ = gen_a_sheet(sender,
                           np.array(receivers),
                           [[] for receiver in range(len(receiver_profiles))],
                           rel_sender2receiver,
                           rel_receiver2sender)

    return sum(sheet['返信有無'])


def online_performance_per_sender_random(sender: int, sender_profiles: np.ndarray, receiver_profiles: np.ndarray,
                                           rel_sender2receiver: np.ndarray, rel_receiver2sender: np.ndarray, candidates: np.ndarray) -> int:
    """
    送信者senderに対してランダムに受信者を表示した時に成立するマッチ数を返す
    @param sender: 送信者
    @param sender_profiles: メッセージ送信源となる性別すべてのユーザのprofile [(ユーザ数, 100)]
    @param receiver_profiles: メッセージ送信先となる性別すべてのユーザのprofile [(ユーザ数, 100)]
    @param rel_sender2receiver: メッセージ送信源の性別から送信先の性別への嗜好度合いが格納された配列
    @param rel_receiver2sender: メッセージ送信先の性別から送信源の性別への嗜好度合いが格納された配列
    @param candidates: 各ユーザについての、まだメッセージを送っていない相手のリスト
    @return マッチ数
    """
    # random推薦を行なった場合
    receivers = random.sample(candidates[sender].tolist(), S)
    sheet, _ = gen_a_sheet(sender,
                           np.array(receivers),
                           [[] for receiver in range(len(receiver_profiles))],
                           rel_sender2receiver,
                           rel_receiver2sender)
    return sum(sheet['返信有無'])


### RCFの実装 ###
def make_messaged(logdata: list, sender_profiles: np.ndarray, receiver_profiles: np.ndarray) -> list:
    """
    各ユーザのメッセージ送信相手に関する情報を持つリストを生成
    @param logdata: メッセージ送信、返信データ
    @param sender_profiles: メッセージ送信源となる性別すべてのユーザのprofile [(ユーザ数, 100)]
    @param receiver_profiles: メッセージ送信先となる性別すべてのユーザのprofile [(ユーザ数, 100)]
    @return users_messaged: メッセージ送信履歴を持つlist. e.g.) iがjにメッセージを送信した時、users_messaged[i]は要素jを持つ
    """
    users_messaged = [[] for user in range(len(sender_profiles))]
    for n in range(len(logdata)):
        for item in logdata[n].itertuples():
            if item.送信有無 == 1:
                users_messaged[item.送信者].append(item.受信者)

    return users_messaged


def calc_attractiveness_similarities(messaged: np.ndarray, sender_profiles: np.ndarray) -> np.ndarray:
    """
    @param messaged: メッセージ送信履歴を持つlist. e.g.) iがjにメッセージを送信した時、users_messaged[i]は要素jを持つ
    @param sender_profiles: メッセージ送信源となる性別すべてのユーザのprofile [(ユーザ数, 100)]
    @param receiver_profiles: メッセージ送信先となる性別すべてのユーザのprofile [(ユーザ数, 100)]
    @return attractiveness_similarityが格納された配列((i, j)成分にはユーザattractiveness_similarity(i, j)が入っている)
    """
    attractiveness_similarities = np.empty((len(sender_profiles), len(sender_profiles)), float)
    for target in range(len(sender_profiles)):
        for other in range(len(sender_profiles)):
            if target != other and len( np.union1d(messaged[target], messaged[other]) ) > 0:
                attractiveness_similarities[target, other] = len( np.intersect1d(messaged[target], messaged[other]) ) / len( np.union1d(messaged[target], messaged[other]) )
                # numerator = len( np.intersect1d(messaged[target], messaged[other]) )
                # denominator = len( np.union1d(messaged[target], messaged[other]) )
                # attractiveness_similarities[target, other] = numerator / denominator

            else:
                attractiveness_similarities[target, other] = 0

    return attractiveness_similarities


def CF1(sender: int, receiver: int, senders_messaged: np.ndarray, receivers_messaged: np.ndarray,
        senders_attractiveness_similarities: list, receivers_attractiveness_similarities: list) -> float:
    """
    @param sender: 送信者を表す整数
    @param receiver: 受信者を表す整数
    @param senders_messaged: 送信者側のメッセージ送信履歴を持つlist. e.g.) iがjにメッセージを送信した時、users_messaged[i]は要素jを持つ
    @param receivers_messaged: 受信者側のメッセージ送信履歴を持つlist. e.g.) iがjにメッセージを送信した時、users_messaged[i]は要素jを持つ
    @param senders_attractiveness_similarities: 送信者同士のattractiveness_similarityを表す配列
    @param receivers_attractiveness_similarities: 受信者同士のattractiveness_similarityを表す配列
    @return 相互スコア
    """
    score1 = 0
    if len(senders_messaged[sender]) > 0:
        for neighbor in senders_messaged[sender]:
            score1 += receivers_attractiveness_similarities[receiver][neighbor]
        score1 /= len(senders_messaged[sender])

    score2 = 0
    if len(receivers_messaged[receiver]) > 0:
        for neighbor in receivers_messaged[receiver]:
            score2 += senders_attractiveness_similarities[sender][neighbor]

        score2 /= len(receivers_messaged[receiver])

    if score1 > 0 and score2 > 0:
        return (2 * score1 * score2) / (score1 + score2)
    else:
        return 0


def online_performance_per_sender_rcf(sender: int, sender_profiles: np.ndarray, receiver_profiles: np.ndarray,
                                      rel_sender2receiver: np.ndarray, rel_receiver2sender: np.ndarray, candidates: np.ndarray,
                                      senders_messaged: np.ndarray, receivers_messaged: np.ndarray,
                                      senders_attractiveness_similarities: list, receivers_attractiveness_similarities: list) -> int:
    """
    送信者senderに対してランダムに受信者を表示した時に成立するマッチ数を返す
    @param sender: 送信者
    @param sender_profiles: メッセージ送信源となる性別すべてのユーザのprofile [(ユーザ数, 100)]
    @param receiver_profiles: メッセージ送信先となる性別すべてのユーザのprofile [(ユーザ数, 100)]
    @param rel_sender2receiver: メッセージ送信源の性別から送信先の性別への嗜好度合いが格納された配列
    @param rel_receiver2sender: メッセージ送信先の性別から送信源の性別への嗜好度合いが格納された配列
    @param candidates: 各ユーザについての、まだメッセージを送っていない相手のリスト
    @param senders_messaged: 送信者側のメッセージ送信履歴を持つlist. e.g.) iがjにメッセージを送信した時、users_messaged[i]は要素jを持つ
    @param receivers_messaged: 受信者側のメッセージ送信履歴を持つlist. e.g.) iがjにメッセージを送信した時、users_messaged[i]は要素jを持つ
    @param senders_attractiveness_similarities: 送信者同士のattractiveness_similarityを表す配列
    @param receivers_attractiveness_similarities: 受信者同士のattractiveness_similarityを表す配列
    @return オンラインテストでのマッチ数
    """
    candidates_with_scores = []
    for candidate in candidates[sender]:
        reciprocal_score = CF1(sender, candidate, senders_messaged, receivers_messaged,
                               senders_attractiveness_similarities, receivers_attractiveness_similarities,)
        candidates_with_scores.append((candidate, reciprocal_score))

    output = sorted(candidates_with_scores, key=lambda x:x[1], reverse=True)
    receivers = [i[0] for i in output[:S]]
    sheet, _ = gen_a_sheet(sender,
                           np.array(receivers),
                           [[] for receiver in range(len(receiver_profiles))],
                           rel_sender2receiver,
                           rel_receiver2sender)
    return sum(sheet['返信有無'])