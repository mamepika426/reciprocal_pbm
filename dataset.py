import numpy as np
import pandas as pd
import torch

from torch import FloatTensor
from torch.utils.data import Dataset

from consts import POW_SEND, POW_REPLY, USE_TRUE_PB


# 参考: https://dreamer-uma.com/pytorch-dataset/
class MyDataset(Dataset):
    def __init__(self, logdata, sender_profiles, receiver_profiles, S):
        # np.emptyで初期化してappend: https://qiita.com/fist0/items/d0779ff861356dafaf95
        features = np.empty((0, S, 200))
        match = np.empty((0, S))
        true_reciprocal_score = np.empty((0, S))
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

            # match作成
            match_sheet = logdata[n]['返信有無'].values
            match_sheet = np.expand_dims(match_sheet, axis=0)
            match = np.append(match, match_sheet, axis=0)
        
            # true_reciprocal_score作成
            true_reciprocal_score_sheet = logdata[n]['gamma_sender'].values * logdata[n]['gamma_receiver'].values
            true_reciprocal_score_sheet = np.expand_dims(true_reciprocal_score_sheet, axis=0)
            true_reciprocal_score = np.append(true_reciprocal_score, true_reciprocal_score_sheet, axis=0)

            if USE_TRUE_PB:
                # theta_send作成
                theta_send_sheet = ( (0.9 / logdata[n]['受信者位置']) ** POW_SEND ).values
                theta_send_sheet = np.expand_dims(theta_send_sheet, axis=0)
                theta_send = np.append(theta_send, theta_send_sheet, axis=0)

                # theta_reply作成
                theta_reply_sheet = ((0.9 / logdata[n]['送信者位置']) ** POW_REPLY).values
                theta_reply_sheet = np.expand_dims(theta_reply_sheet, axis=0)
                theta_reply = np.append(theta_reply, theta_reply_sheet, axis=0)

        self.features = features
        self.match = match
        self.true_reciprocal_score = true_reciprocal_score
        self.theta_send = theta_send
        self.theta_reply = theta_reply
     
    def __len__(self):
         return self.features.shape[0]

    def __getitem__(self, idx):
        data_dict = {'features': torch.FloatTensor(self.features[idx]),
                     'match': torch.FloatTensor(self.match[idx]),
                     'true_reciprocal_score': torch.FloatTensor(self.true_reciprocal_score[idx]),
                     'pscore': torch.FloatTensor( self.theta_send[idx] * self.theta_reply[idx] )}
        return data_dict
