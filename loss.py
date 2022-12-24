from typing import Optional

from torch import ones, FloatTensor
from torch.nn.functional import log_softmax


def listwise_loss(scores: FloatTensor, implicit_feedback: FloatTensor, pscore: Optional[FloatTensor] = None) -> FloatTensor:
    """
    リストワイズ損失
    @param socres: スコアリング関数の出力 [(train_batch_size, slate_size)]
    @param implicit_feedback: ラベル [(train_batch_size, slate_size)]
    @param pscore: 傾向スコア [(train_batch_size, slate_size)]
    """
    if pscore is None:
        pscore = ones(implicit_feedback.shape[1])
    listwise_loss = 0
    for scores_, implicit_feedback_, pscore_ in zip(scores, implicit_feedback, pscore):
        listwise_loss_ = (implicit_feedback_ / pscore_) * log_softmax(scores_, dim=0)
        listwise_loss -= listwise_loss_.sum()
    return listwise_loss / len(scores)
