import numpy as np

from torch import LongTensor, nn
from torch.utils.data import DataLoader
from pytorchltr.evaluation.dcg import ndcg

from consts import S
from dataset import MyDataset

def evaluate_test_performance(score_fn: nn.Module, test: MyDataset, S: int, test_batch_size: int) -> float:
    """
    与えられたmodelのランキング性能をテストデータにおける真の嗜好度合情報をつかってnDCG@10で評価する.
    """
    loader = DataLoader(test, test_batch_size, shuffle=False, drop_last=True)
    ndcg_score = 0.0
    n = LongTensor(np.ones(test_batch_size)* (S-1))
    for batch in loader:
        true_reciprocal_score = batch['true_reciprocal_score']
        ndcg_score += ndcg(
            score_fn(batch['features']), true_reciprocal_score, n, k=10, exp=False
        ).sum()

    return float(ndcg_score / len(test))
