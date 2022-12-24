from typing import Optional

import numpy as np
import pandas as pd
import torch

from torch import FloatTensor
from torch.utils.data import Dataset

from consts import POW_SEND, POW_REPLY, USE_TRUE_PB

class BaseDataset(Dataset):
    def __init__(self, logdata: list, sender_profiles: np.ndarray, receiver_profiles: np.ndarray,
                 S: int, theta_send_est: Optional[np.ndarray] = None, theta_reply_est: Optional[np.ndarray] = None) -> None:
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


# 参考: https://dreamer-uma.com/pytorch-dataset/
class DatasetBothSide(BaseDataset):
    def __init__(self, logdata: list, sender_profiles: np.ndarray, receiver_profiles: np.ndarray,
                 S: int, theta_send_est: Optional[np.ndarray] = None, theta_reply_est: Optional[np.ndarray] = None) -> None:
        # np.emptyで初期化してappend: https://qiita.com/fist0/items/d0779ff861356dafaf95
        features = np.empty((0, S, 200))
        implicit_feedback = np.empty((0, S))
        true_score = np.empty((0, S))
        theta_send = np.empty((0, S))
        theta_reply = np.empty((0, S))

        for n in range(len(logdata)):
            # features作成
            features_sheet = np.empty((0, 200), float)
            # indexいらないならiterrows使うな(intがfloatになるなど): https://biotech-lab.org/articles/10669
            for item in logdata[n].itertuples():
                features_row =  np.hstack( (sender_profiles[item.送信者], receiver_profiles[item.受信者]) )
                # 次元を追加してappend: https://www.delftstack.com/ja/howto/numpy/python-numpy-add-dimension/
                features_row = np.expand_dims(features_row, axis=0)
                features_sheet = np.append(features_sheet, features_row, axis=0)
        
            features_sheet = np.expand_dims(features_sheet, axis=0)
            features = np.append(features, features_sheet, axis=0)

            # implicit_feedback作成
            implicit_feedback_sheet = logdata[n]['返信有無'].values
            implicit_feedback_sheet = np.expand_dims(implicit_feedback_sheet, axis=0)
            implicit_feedback = np.append(implicit_feedback, implicit_feedback_sheet, axis=0)
        
            # true_score作成
            true_score_sheet = logdata[n]['gamma_sender'].values * logdata[n]['gamma_receiver'].values
            true_score_sheet = np.expand_dims(true_score_sheet, axis=0)
            true_score = np.append(true_score, true_score_sheet, axis=0)

            # ポジションバイアスの推定値が与えられていない場合は真のポジションバイアスで補正
            if theta_send_est is None and theta_reply_est is None:
                # theta_send作成
                theta_send_sheet = ((0.9 / logdata[n]['受信者位置']) ** POW_SEND).values
                theta_send_sheet = np.expand_dims(theta_send_sheet, axis=0)
                theta_send = np.append(theta_send, theta_send_sheet, axis=0)

                # theta_reply作成
                theta_reply_sheet = ((0.9 / logdata[n]['送信者位置']) ** POW_REPLY).values
                theta_reply_sheet = np.expand_dims(theta_reply_sheet, axis=0)
                theta_reply = np.append(theta_reply, theta_reply_sheet, axis=0)

            else:
                # theta_send作成
                theta_send_sheet = theta_send_est[logdata[n]['受信者位置']]
                theta_send_sheet = np.expand_dims(theta_send_sheet, axis=0)
                theta_send = np.append(theta_send, theta_send_sheet, axis=0)

                # theta_reply作成
                theta_reply_sheet = theta_reply_est[logdata[n]['送信者位置']]
                theta_reply_sheet = np.expand_dims(theta_reply_sheet, axis=0)
                theta_reply = np.append(theta_reply, theta_reply_sheet, axis=0)

        self.features = features
        self.implicit_feedback = implicit_feedback
        self.true_score = true_score
        self.theta_send = theta_send
        self.theta_reply = theta_reply
     
    def __len__(self):
         return self.features.shape[0]

    def __getitem__(self, idx):
        data_dict = {'features': torch.FloatTensor(self.features[idx]),
                     'implicit_feedback': torch.FloatTensor(self.implicit_feedback[idx]),
                     'true_score': torch.FloatTensor(self.true_score[idx]),
                     'pscore': torch.FloatTensor( self.theta_send[idx] * self.theta_reply[idx] )}
        return data_dict


class DatasetOneSide(BaseDataset):
    def __init__(self, logdata: list, sender_profiles: np.ndarray, receiver_profiles: np.ndarray,
                 S: int, theta_send_est: Optional[np.ndarray] = None, theta_reply_est: Optional[np.ndarray] = None) -> None:
        # np.emptyで初期化してappend: https://qiita.com/fist0/items/d0779ff861356dafaf95
        features = np.empty((0, S, 200))
        implicit_feedback = np.empty((0, S))
        true_score = np.empty((0, S))
        theta_send = np.empty((0, S))

        for n in range(len(logdata)):
            # features作成
            features_sheet = np.empty((0, 200), float)
            # indexいらないならiterrows使うな(intがfloatになるなど): https://biotech-lab.org/articles/10669
            for item in logdata[n].itertuples():
                features_row =  np.hstack( (sender_profiles[item.送信者], receiver_profiles[item.受信者]) )
                # 次元を追加してappend: https://www.delftstack.com/ja/howto/numpy/python-numpy-add-dimension/
                features_row = np.expand_dims(features_row, axis=0)
                features_sheet = np.append(features_sheet, features_row, axis=0)

            features_sheet = np.expand_dims(features_sheet, axis=0)
            features = np.append(features, features_sheet, axis=0)

            # implicit_feedback作成
            implicit_feedback_sheet = logdata[n]['送信有無'].values
            implicit_feedback_sheet = np.expand_dims(implicit_feedback_sheet, axis=0)
            implicit_feedback = np.append(implicit_feedback, implicit_feedback_sheet, axis=0)

            # true_score作成
            true_score_sheet = logdata[n]['gamma_sender'].values
            true_score_sheet = np.expand_dims(true_score_sheet, axis=0)
            true_score = np.append(true_score, true_score_sheet, axis=0)

            # ポジションバイアスの推定値が与えられていない場合は真のポジションバイアスで補正
            if theta_send_est is None:
                # theta_send作成
                theta_send_sheet = ((0.9 / logdata[n]['受信者位置']) ** POW_SEND).values
                theta_send_sheet = np.expand_dims(theta_send_sheet, axis=0)
                theta_send = np.append(theta_send, theta_send_sheet, axis=0)

            else:
                # theta_send作成
                theta_send_sheet = theta_send_est[logdata[n]['受信者位置']]
                theta_send_sheet = np.expand_dims(theta_send_sheet, axis=0)
                theta_send = np.append(theta_send, theta_send_sheet, axis=0)

        self.features = features
        self.implicit_feedback = implicit_feedback
        self.true_score = true_score
        self.theta_send = theta_send

    def __len__(self):
         return self.features.shape[0]

    def __getitem__(self, idx):
        data_dict = {'features': torch.FloatTensor(self.features[idx]),
                     'implicit_feedback': torch.FloatTensor(self.implicit_feedback[idx]),
                     'true_score': torch.FloatTensor(self.true_score[idx]),
                     'pscore': torch.FloatTensor(self.theta_send[idx])}
        return data_dict
