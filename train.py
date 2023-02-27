# ランキング学習を行い、その性能を評価する関数群
from typing import List

import torch
from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm # プログレスバーの表示

from model import MLPScoreFunc
from consts import S, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, HIDDEN_LAYER_SIZES, LEARNING_RATE, N_EPOCHS
from evaluate import evaluate_test_performance
from loss import listwise_loss
from dataset import BaseDataset


def train_ranker(score_fn: nn.Module, optimizer: optim, estimator: str, train: BaseDataset, test: BaseDataset,
    train_batch_size: int, test_batch_size: int, n_epochs: int) -> List:
    """
    ランキングモデルを学習するための関数.
    @param score_fn: スコアリング関数.
    @param optimizer: パラメータ最適化アルゴリズム.
    @param estimator:
        スコアリング関数を学習するための目的関数を観測データから近似する推定量.
        'naive', 'ips', 'ideal'のいずれかしか与えることができない.
        'ideal'が与えられた場合は、真のreciprocal scoreをもとに、ランキングモデルを学習する.
    @param train: トレーニングデータ.
    @param test:  テストデータ.
    @param train_batch_size: 訓練バッチサイズ.
    @param test_batch_size: テストバッチサイズ
    @param n_epochs: エポック数.
    @return テストデータに対するnDCG@10の値のリスト
    """
    assert estimator in [
        "naive",
        "ips",
        "ideal",
    ], f"estimator must be 'naive', 'ips', or 'ideal', but {estimator} is given"

    ndcg_score_list = list()
    for _ in tqdm(range(n_epochs)):
        loader = DataLoader(
            train,
            batch_size=train_batch_size,
            shuffle=False,
        )
        # トレーニングモード
        score_fn.train()
        for batch in loader:
            if estimator == "naive":
                loss = listwise_loss(
                    scores=score_fn(batch['features']),
                    implicit_feedback=batch['implicit_feedback']
                )
            elif estimator == "ips":
                loss = listwise_loss(
                    scores=score_fn(batch['features']),
                    implicit_feedback=batch['implicit_feedback'],
                    pscore=batch['pscore'],
                )
            elif estimator == "ideal":
                loss = listwise_loss(
                    scores=score_fn(batch['features']),
                    implicit_feedback=batch['true_score']
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        score_fn.eval()
        ndcg_score = evaluate_test_performance(score_fn=score_fn, test=test, S=S, test_batch_size=test_batch_size)
        ndcg_score_list.append(ndcg_score)

    return ndcg_score_list


def train_3rankers(train: BaseDataset, test: BaseDataset) -> dict:
    """
    naive, ips, idealの3パターンの学習を行い、学習後のスコアリングモデルと学習曲線を持つdictを返す
    @param train: トレーニングデータ
    @param test: テストデータ
    @return 学習後のスコアリングモデルと学習曲線を持つdict
    """
    torch.manual_seed(12345)

    score_fn_naive = MLPScoreFunc(input_size=200, hidden_layer_sizes=HIDDEN_LAYER_SIZES)
    optimizer = Adam(score_fn_naive.parameters(), lr=LEARNING_RATE)
    ndcg_score_list_naive = train_ranker(
        score_fn=score_fn_naive,
        optimizer=optimizer,
        estimator="naive",
        train=train,
        test=test,
        train_batch_size=TRAIN_BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE,
        n_epochs=N_EPOCHS
    )

    score_fn_ips = MLPScoreFunc(input_size=200, hidden_layer_sizes=HIDDEN_LAYER_SIZES)
    optimizer = Adam(score_fn_ips.parameters(), lr=LEARNING_RATE)
    ndcg_score_list_ips = train_ranker(
        score_fn=score_fn_ips,
        optimizer=optimizer,
        estimator="ips",
        train=train,
        test=test,
        train_batch_size=TRAIN_BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE,
        n_epochs=N_EPOCHS,
    )

    score_fn_ideal = MLPScoreFunc(input_size=200, hidden_layer_sizes=HIDDEN_LAYER_SIZES)
    optimizer = Adam(score_fn_ideal.parameters(), lr=LEARNING_RATE)
    ndcg_score_list_ideal = train_ranker(
        score_fn=score_fn_ideal,
        optimizer=optimizer,
        estimator="ideal",
        train=train,
        test=test,
        train_batch_size=TRAIN_BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE,
        n_epochs=N_EPOCHS,
    )

    results_dict = {'score_fn_naive': score_fn_naive,
                    'ndcg_score_list_naive': ndcg_score_list_naive,
                    'score_fn_ips': score_fn_ips,
                    'ndcg_score_list_ips': ndcg_score_list_ips,
                    'score_fn_ideal': score_fn_ideal,
                    'ndcg_score_list_ideal': ndcg_score_list_ideal}
    return results_dict
