import sys

import numpy as np
import pandas as pd

from typing import List, Tuple
from scipy.stats import bernoulli
from sklearn.ensemble import GradientBoostingClassifier


# regressionEM_send内部で用いる関数を定義
def np_log(x: np.ndarray) -> np.ndarray:
    """
    float型では、(大体)1e-323未満の値は精度の問題で0.0となり
    対数を取ると -inf となるので下限を抑えておく
    @param x: 実数
    @return : log(x)
    """
    return np.log(np.clip(a=x, a_min=1e-323, a_max=1e+10))


def calculate_log_likelihood(logdata: list, theta_est: np.ndarray, gamma_est: np.ndarray) -> float:
    """
    対数尤度を計算する関数
    @param logdata: 送信、返信ログ
    @param theta_est: 推定されたポジションバイアス [(slate_size + 1)]
    @param gamma_est: 推定された嗜好度合い [(メッセージ送信側の性別のユーザ数, メッセージ受信側の性別のユーザ数)]
    @return log_likelihood: 対数尤度
    """
    log_likelihood = 0
    for n in range(len(logdata)):
        for item in logdata[n].itertuples():
            theta_gamma = theta_est[ item.受信者位置 ] * gamma_est[ item.送信者, item.受信者 ]
            log_likelihood += item.送信有無 * np_log(theta_gamma) + (1 - item.送信有無) * np_log(1 - theta_gamma)

    return log_likelihood


def Estep(logdata: list, num_senders: int, num_receivers: int, S: int, theta_est: np.ndarray, gamma_est: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    E stepを実行する関数
    @param logdata: 送信、返信ログ
    @param num_senders: メッセージ送信側の性別のユーザ数
    @param num_receivers: メッセージ受信側の性別のユーザ数
    @param S: slate_size
    @param theta_est: 推定されたポジションバイアス [(slate_size + 1)]
    @param gamma_est: 推定された嗜好度合い [(num_senders, num_receivers)]
    @return P_O: Pr(O(k)=1 | S(sender, receiver, k)=0)の推定値が格納された配列 [(num_senders, num_receivers, slate_size + 1)]
    @return P_R: Pr(P(sender, receiver)=1 | S(sender, receiver, k)=0)の推定値が格納された配列 [(num_senders, num_receivers, slate_size + 1)]
    """
    P_O = np.empty((num_senders, num_receivers, S+1))
    P_R = np.empty((num_senders, num_receivers, S+1))
    for n in range(len(logdata)):
        for item in logdata[n].itertuples():
            if item.送信有無 == 0:
                theta_numerator = theta_est[item.受信者位置] * (1 - gamma_est[ item.送信者, item.受信者 ])
                gamma_numerator = gamma_est[ item.送信者, item.受信者 ] * (1 - theta_est[item.受信者位置])
                denominator = 1 - theta_est[item.受信者位置] * gamma_est[ item.送信者, item.受信者 ] + 1e-9 # 0割り対策
                P_O[item.送信者, item.受信者, item.受信者位置] = theta_numerator / denominator
                P_R[item.送信者, item.受信者, item.受信者位置] = gamma_numerator / denominator

    return P_O, P_R


def sampling_relevances(logdata: list, P_R: np.ndarray, S: int) -> np.ndarray:
    """
    推定されたパラメータをもとにして嗜好度合いについてサンプリング
    @param logdata: 送信、返信ログ
    @param P_R: Pr(P(x, y)=1 | S(sender, receiver, k)=0)の推定値が格納された配列 [(num_senders, num_receivers, slate_size + 1)]
    @param S: slate size
    @return sampled_relevances: サンプリングされた嗜好度合い
    """
    sampled_relevances = np.empty(0)
    random_state = 0
    while(random_state < 100):
        for n in range(len(logdata)):
            for item in logdata[n].itertuples():
                if item.送信有無 == 1:
                    sampled_relevances = np.append(sampled_relevances, 1)
                else:
                    # 乱数にitem.受信者を足しているのはシート内で乱数を固定させないため
                    sampled_relevances = np.append(sampled_relevances, 
                                         bernoulli.rvs(P_R[item.送信者, item.受信者, item.受信者位置], random_state=random_state + item.受信者) )
        
        # サンプリング後の嗜好度合いラベルが片方のみの場合、エラーになるのでリトライ
        if len(set(sampled_relevances)) <= 1:
            random_state += 1           
        else:
            break
    # 100回リトライしてもサンプリングできなければ強制終了
    if random_state == 99:
        sys.exit("サンプリングがうまくいきませんでした")
    else:
        return sampled_relevances
    

def update_theta(logdata: list, P_O: np.ndarray, S: int) -> np.ndarray:
    """
    M stepでthetaのパラメータを更新
    @param logdata: 送信、返信ログ
    @param P_O: Pr(O(k)=1 | S(*, *', k)=0)の推定値が格納された配列 [(slate_size)]
    @param S: slate size
    @return theta_est: 更新後の推定されたポジションバイアス [(slate_size + 1)]
    """
    theta_est = np.zeros(S+1)
    for n in range(len(logdata)):
        for item in logdata[n].itertuples():
            theta_est[item.受信者位置] += item.送信有無 + (1 - item.送信有無) * P_O[item.送信者, item.受信者, item.受信者位置]
    
    return theta_est / len(logdata)


def update_gamma(logdata: list, sender_profiles: np.ndarray, receiver_profiles: np.ndarray, sampled_relevances: np.ndarray) -> np.ndarray:
    """
    M stepでgammaのパラメータを更新
    @param logdata: 送信、返信ログ
    @param sender_profiles: メッセージ送信先である性別すべてのユーザのprofile [(ユーザ数, 100)]
    @param receiver_profiles: メッセージ送信先となる性別すべてのユーザのprofile [(ユーザ数, 100)]
    @param sampled_relevances: サンプリングされた嗜好度合い
    @return gamma_est: 更新後の推定された嗜好度合い [(メッセージ送信側の性別のユーザ数, メッセージ受信側の性別のユーザ数)]
    """
    # (送信者, 受信者)ペアの特徴量作成
    features = np.empty((0, 200), float)
    for n in range(len(logdata)):
        # features作成
        for item in logdata[n].itertuples():
            features_pair =  np.hstack( (sender_profiles[item.送信者], receiver_profiles[item.受信者]) )
            # 次元を追加してappend: https://www.delftstack.com/ja/howto/numpy/python-numpy-add-dimension/
            features_pair = np.expand_dims(features_pair, axis=0)
            features = np.append(features, features_pair, axis=0)
    
    # GBDT学習
    gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, max_depth=3, random_state=0)
    gbdt.fit(features, sampled_relevances)
    
    # gamma更新
    gamma_est = np.empty((0, len(receiver_profiles)))
    for sender in range(len(sender_profiles)):
        # np.tileで同じ配列を縦に繰り返し: https://note.nkmk.me/python-numpy-tile/
        sender_repeat = np.tile(sender_profiles[sender], (len(receiver_profiles), 1)) # [len(receiver_profiles), 100]
        feature_a_sender = np.hstack((sender_repeat, receiver_profiles))              # [len(receiver_profiles), 200] 
        gamma_est_a_sender = gbdt.predict_proba(feature_a_sender)[:, 1]
        gamma_est_a_sender = np.expand_dims(gamma_est_a_sender, axis=0)
        gamma_est = np.append(gamma_est, gamma_est_a_sender, axis=0)      # [len(sender_profiles), len(receiver_profiles)]

    return gamma_est


def regressionEM_send(logdata: list, sender_profiles: np.ndarray, receiver_profiles: np.ndarray, S: int, max_iter: int, threshold: float) -> Tuple[np.ndarray, list]:
    """
    EMアルゴリズムによるポジションバイアスの推定を行う関数
    @param logdata: 送信、返信ログ
    @param sender_profiles: メッセージ送信先である性別すべてのユーザのprofile [(ユーザ数, 100)]
    @param receiver_profiles: メッセージ送信先となる性別すべてのユーザのprofile [(ユーザ数, 100)]
    @param S: slate size
    @param max_iter: 最大反復回数
    @param threshold: 対数尤度の更新がこれより小さくなると終了
    @return theta_est: 推定されたポジションバイアス [(slate_size)]
    @return log_likelihood_list: イテレーションのたびに記録された対数尤度
    """
    # 各パラメータを0.5で初期化
    theta_est = np.ones(S+1) / 2
    gamma_est = np.ones([len(sender_profiles), len(receiver_profiles)]) / 2

    # 対数尤度を計算
    log_likelihood = calculate_log_likelihood(logdata, theta_est, gamma_est)

    log_likelihood_list = []
    for i in range(1, max_iter + 1):
        print("ループ{0}回目".format(i))
        # E step(estimate the hidden variable probabilities)
        P_O, P_R = Estep(logdata, len(sender_profiles), len(receiver_profiles), S, theta_est, gamma_est)
        # generate training data for GBDT
        sampled_relevances = sampling_relevances(logdata, P_R, S)
        # M step(update theta and gamma)
        theta_est = update_theta(logdata, P_O, S)
        print(theta_est)
        gamma_est = update_gamma(logdata, sender_profiles, receiver_profiles, sampled_relevances)
        
        # 対数尤度を計算
        new_log_likelihood = calculate_log_likelihood(logdata, theta_est, gamma_est)
        log_likelihood_list.append(new_log_likelihood)

        # 収束判定
        if abs(log_likelihood - new_log_likelihood) < threshold:
            break

    return theta_est, log_likelihood_list


# regressionEM_reply内部で用いる関数を定義
def calculate_log_likelihood_reply(logdata_extracted: list, theta_est: np.ndarray, gamma_est: np.ndarray) -> float:
    """
    返信ログに対する対数尤度を計算する関数
    @param logdata_extracted: `送信有無==1`のものだけ抽出した送信、返信ログ
    @param theta_est: 推定された返信時のポジションバイアス [(max_sender_position + 1)]
    @param gamma_est: 推定された返信時の嗜好度合い [(メッセージ返信側の性別のユーザ数, メッセージ送信側の性別のユーザ数)]
    @return log_likelihood: 対数尤度
    """
    log_likelihood = 0
    for n in range(len(logdata_extracted)):
        for item in logdata_extracted[n].itertuples():
            theta_gamma = theta_est[ item.送信者位置 ] * gamma_est[ item.受信者, item.送信者 ]
            log_likelihood += item.返信有無 * np_log(theta_gamma) + (1 - item.返信有無) * np_log(1 - theta_gamma)

    return log_likelihood


def Estep_reply(logdata_extracted: list, num_senders: int, num_receivers: int, max_sender_position: int, theta_est: np.ndarray, gamma_est: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    E stepを実行する関数
    @param logdata_extracted: `送信有無==1`のものだけ抽出した送信、返信ログ
    @param num_senders: メッセージ送信側の性別のユーザ数
    @param num_receivers: メッセージ受信側の性別のユーザ数
    @param max_sender_position: 送信者位置の最大値
    @param theta_est: 推定されたポジションバイアス [(max_sender_position + 1)]
    @param gamma_est: 推定された嗜好度合い [(num_receivers, num_senders)]
    @return P_O: Pr(O'(k')=1 | R(receiver, sender, k')=0)の推定値が格納された配列 [(num_receivers, num_senders, max_sender_position + 1)]
    @return P_R: Pr(P(receiver, sender)=1 | R(receiver, sender, k')=0)の推定値が格納された配列 [(num_receivers, num_senders, max_sender_position + 1)]
    """
    P_O = np.empty((num_receivers, num_senders, max_sender_position+1))
    P_R = np.empty((num_receivers, num_senders, max_sender_position+1))
    for n in range(len(logdata_extracted)):
        for item in logdata_extracted[n].itertuples():
            if item.返信有無 == 0:
                theta_numerator = theta_est[item.送信者位置] * (1 - gamma_est[ item.受信者, item.送信者 ])
                gamma_numerator = gamma_est[ item.受信者, item.送信者 ] * (1 - theta_est[item.送信者位置])
                denominator = 1 - theta_est[item.送信者位置] * gamma_est[ item.受信者, item.送信者 ] + 1e-9 # 0割り対策
                P_O[item.受信者, item.送信者, item.送信者位置] = theta_numerator / denominator
                P_R[item.受信者, item.送信者, item.送信者位置] = gamma_numerator / denominator

    return P_O, P_R


def sampling_relevances_reply(logdata_extracted: list, P_R: np.ndarray, max_sender_position: int) -> np.ndarray:
    """
    推定されたパラメータをもとにして嗜好度合いについてサンプリング
    @param logdata_extracted: `送信有無==1`のものだけ抽出した送信、返信ログ
    @param P_R: Pr(P(receiver, sender)=1 | R(receiver, sender, k')=0)の推定値が格納された配列 [(num_receivers, num_senders, max_sender_position + 1)]
    @param max_sender_position: 送信者位置の最大値
    @return sampled_relevances: サンプリングされた嗜好度合い
    """
    sampled_relevances = np.empty(0)
    random_state = 0
    while(random_state < 100):
        for n in range(len(logdata_extracted)):
            for item in logdata_extracted[n].itertuples():
                if item.返信有無 == 1:
                    sampled_relevances = np.append(sampled_relevances, 1)
                else:
                    # 乱数にitem.受信者を足しているのはシート内で乱数を固定させないため
                    sampled_relevances = np.append(sampled_relevances, 
                                         bernoulli.rvs(P_R[item.受信者, item.送信者, item.送信者位置], random_state=random_state + item.受信者) )
                    
        # サンプリング後の嗜好度合いラベルが片方のみの場合、エラーになるのでリトライ
        if len(set(sampled_relevances)) <= 1:
            random_state += 1           
        else:
            break
    if random_state == 99:
        sys.exit("サンプリングがうまくいきませんでした")
    else:
        return sampled_relevances


def update_theta_reply(logdata_extracted: list, P_O: np.ndarray, max_sender_position: int) -> np.ndarray:
    """
    M stepでthetaのパラメータを更新
    @param logdata_extracted: `送信有無==1`のものだけ抽出した送信、返信ログ
    @param P_O: Pr(O'(k')=1 | R(receiver, sender, k')=0)の推定値が格納された配列 [(num_receivers, num_senders, max_sender_position + 1)]
    @param max_sender_position: 送信者位置の最大値
    @return theta_est: 更新後の推定された返信時のポジションバイアス [(max_sender_position + 1)]
    """
    theta_est = np.zeros(max_sender_position+1)
    # `送信者位置`カラムの値ごとにログデータ中に現れた行数をカウント
    cnt_array = np.zeros(max_sender_position+1)
    cnt_array[0] = 1 # 0割り防止
    for n in range(len(logdata_extracted)):
        for item in logdata_extracted[n].itertuples():
            theta_est[item.送信者位置] += item.返信有無 + (1 - item.返信有無) * P_O[item.受信者, item.送信者, item.送信者位置]
            cnt_array[item.送信者位置] += 1
    
    return theta_est / cnt_array


def update_gamma_reply(logdata_extracted: list, sender_profiles: np.ndarray, receiver_profiles: np.ndarray, sampled_relevances: np.ndarray) -> np.ndarray:
    """
    M stepでgammaのパラメータを更新
    @param ogdata_extracted: `送信有無==1`のものだけ抽出した送信、返信ログ
    @param sender_profiles: メッセージ送信先である性別すべてのユーザのprofile [(ユーザ数, 100)]
    @param receiver_profiles: メッセージ送信先となる性別すべてのユーザのprofile [(ユーザ数, 100)]
    @param sampled_relevances: サンプリングされた嗜好度合い
    @return gamma_est: 更新後の推定された嗜好度合い [(メッセージ送信側の性別のユーザ数, メッセージ受信側の性別のユーザ数)]
    """
    # (送信者, 受信者)ペアの特徴量作成
    features = np.empty((0, 200), float)
    for n in range(len(logdata_extracted)):
        # features作成
        for item in logdata_extracted[n].itertuples():
            features_pair =  np.hstack( (receiver_profiles[item.受信者], sender_profiles[item.送信者]) )
            # 次元を追加してappend: https://www.delftstack.com/ja/howto/numpy/python-numpy-add-dimension/
            features_pair = np.expand_dims(features_pair, axis=0)
            features = np.append(features, features_pair, axis=0)
    
    # GBDT学習
    gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, max_depth=3, random_state=0)
    gbdt.fit(features, sampled_relevances)
    
    # gamma更新
    gamma_est = np.empty((0, len(sender_profiles)))
    for receiver in range(len(receiver_profiles)):
        # np.tileで同じ配列を縦に繰り返し: https://note.nkmk.me/python-numpy-tile/
        receiver_repeat = np.tile(receiver_profiles[receiver], (len(sender_profiles), 1)) # [len(sender_profiles), 100]
        feature_a_receiver = np.hstack((receiver_repeat, sender_profiles))              # [len(sender_profiles), 200] 
        gamma_est_a_receiver = gbdt.predict_proba(feature_a_receiver)[:, 1]
        gamma_est_a_receiver = np.expand_dims(gamma_est_a_receiver, axis=0)
        gamma_est = np.append(gamma_est, gamma_est_a_receiver, axis=0)      # [len(sender_profiles), len(receiver_profiles)]

    return gamma_est


def regressionEM_reply(logdata: list, sender_profiles: np.ndarray, receiver_profiles: np.ndarray, max_iter: int, threshold: float) -> Tuple[np.ndarray, list, int]:
    """
    EMアルゴリズムによるポジションバイアスの推定を行う関数
    @param logdata: 送信、返信ログ
    @param sender_profiles: メッセージ送信先である性別すべてのユーザのprofile [(ユーザ数, 100)]
    @param receiver_profiles: メッセージ送信先となる性別すべてのユーザのprofile [(ユーザ数, 100)]
    @param max_iter: 最大反復回数
    @param threshold: 対数尤度の更新がこれより小さくなると終了
    @return theta_est: 推定された返信時のポジションバイアス [(max_sender_position+1)]
    @return log_likelihood_list: イテレーションのたびに記録された対数尤度
    @return max_sender_position: ログデータに現れる送信者位置の最大値
    """
    # 送信者位置の最大値を取得
    max_sender_position = 1
    for n in range(len(logdata)):
        for item in logdata[n].itertuples():
            if item.送信有無 == 1 and max_sender_position < item.送信者位置:
                max_sender_position = item.送信者位置
                
    # 送信有無 == 1のログだけ抜き出す
    logdata_extracted = []
    for n in range(len(logdata)):
        logdata_extracted.append(logdata[n][logdata[n]['送信有無']==1])

    # 各パラメータを0.5で初期化
    theta_est = np.ones(max_sender_position+1) / 2
    gamma_est = np.ones([len(receiver_profiles), len(sender_profiles)]) / 2
    
    # 対数尤度を計算
    log_likelihood = calculate_log_likelihood_reply(logdata_extracted, theta_est, gamma_est)

    log_likelihood_list = []
    for i in range(1, max_iter + 1):
        print("ループ{0}回目".format(i))
        # E step(estimate the hidden variable probabilities)
        P_O, P_R = Estep_reply(logdata_extracted, len(sender_profiles), len(receiver_profiles), max_sender_position, theta_est, gamma_est)
        # generate training data for GBDT
        sampled_relevances = sampling_relevances_reply(logdata_extracted, P_R, max_sender_position)
        # M step(update theta and gamma)
        theta_est = update_theta_reply(logdata_extracted, P_O, max_sender_position)
        gamma_est = update_gamma_reply(logdata_extracted, sender_profiles, receiver_profiles, sampled_relevances)
        print(theta_est)
        # 対数尤度を計算
        new_log_likelihood = calculate_log_likelihood_reply(logdata_extracted, theta_est, gamma_est)
        log_likelihood_list.append(new_log_likelihood)

        # 収束判定
        if abs(log_likelihood - new_log_likelihood) < threshold:
            break

    return theta_est, log_likelihood_list, max_sender_position
