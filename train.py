from typing import List, Optional

from torch import nn, optim
from torch.utils .data import DataLoader
from tqdm import tqdm # プログレスバーの表示
from dataset import MyDataset

from consts import S
from evaluate import evaluate_test_performance
from loss import listwise_loss

def train_ranker(score_fn: nn.Module, optimizer: optim, estimator: str, train: MyDataset, test: MyDataset,
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
        cnt = 0
        for batch in loader:
            if estimator == "naive":
                loss = listwise_loss(
                    scores=score_fn(batch['features']),
                    match=batch['match']
                )
            elif estimator == "ips":
                loss = listwise_loss(
                    scores=score_fn(batch['features']),
                    match=batch['match'],
                    pscore=batch['pscore'],
                )
            elif estimator == "ideal":
                loss = listwise_loss(
                    scores=score_fn(batch['features']),
                    match=batch['true_reciprocal_score']
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        score_fn.eval()
        ndcg_score = evaluate_test_performance(score_fn=score_fn, test=test, S=S, test_batch_size=test_batch_size)
        ndcg_score_list.append(ndcg_score)

    return ndcg_score_list