import os

# 環境
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 実行ファイルがあるディレクトリ
DATA_DIR = os.path.join(BASE_DIR, "data") # 人工データを保存するディレクトリ
DATA_DIR_M2F = os.path.join(DATA_DIR, "m2f")  # 男性から女性に送信したメッセージについてのログを保存するディレクトリ
DATA_DIR_F2M = os.path.join(DATA_DIR, "f2m")  # 女性から男性に送信したメッセージについてのログを保存するディレクトリ
FIG_DIR = os.path.join(BASE_DIR, "fig") # 実験結果の画像を保存するディレクトリ

# 男女の数を設定
NUM_MALES = 1000
NUM_FEMALES = 1200

# クラスター数設定
NUM_CLUSTERS = 10

# slate_size(1シートに表示する人数)
S = 10
NUM_SHEETS_M2F = 3000  # 男性から女性に送信したメッセージについてのログの総数
NUM_SHEETS_F2M = 3000  # 女性から男性に送信したメッセージについてのログの総数

NUM_TRAIN_SHEETS_M2F = 2500 # トレーニングに回すシート枚数(男性から女性に送信したメッセージについてのログ)
NUM_TRAIN_SHEETS_F2M = 2500 # トレーニングに回すシート枚数(女性から男性に送信したメッセージについてのログ)

# ポジションバイアスを推定する際に, サンプル数が少なすぎる(< MIN_SAMPLES_FOR_EST)
# '送信者位置'に対しては ポジションバイアス = EXCEPTIONAL_PB_VALUE で対応する
MIN_SAMPLES_FOR_EST = 30
EXCEPTIONAL_PB_VALUE = 0.1

# ポジションバイアス推定のためのEMアルゴリズムのループ数、閾値設定
MAX_ITER_SEND = 5
MAX_ITER_REPLY = 10
THRESHOLD_SEND = 1
THRESHOLD_REPLY = 1

# ポジションバイアス強度設定
POW_SEND = 1
POW_REPLY = 1

# ニューラルネットに関するパラメータ設定
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
HIDDEN_LAYER_SIZES = (10, 10)
LEARNING_RATE = 0.0001
N_EPOCHS = 100

# 補正値に真の値を用いるか推定されたバイアスを用いるか
USE_TRUE_PB = True
