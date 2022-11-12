import os

# 環境
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 実行ファイルがあるディレクトリ
DATA_DIR = os.path.join(BASE_DIR, "data") # 人工データを保存するディレクトリ
FIG_DIR = os.path.join(BASE_DIR, "fig") # 実験結果の画像を保存するディレクトリ

# 男女の数を2000人に設定
NUM_MALES = 400
NUM_FEMALES = 500

# クラスター数設定
NUM_CLUSTERS = 10

# slate_size(1シートに表示する人数), 検索総回数設定
S = 10
NUM_SHEETS = 1000
NUM_TRAIN_SHEETS = 800

# ポジションバイアス強度設定
POW_SEND = 1
POW_REPLY = 1

# ニューラルネットに関するパラメータ設定
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
HIDDEN_LAYER_SIZES = (10, 10)
LEARNING_RATE = 0.0001
N_EPOCHS = 100

# 補正値に真の値を用いるか推定されたバイアスを用いるかどうか
USE_TRUE_PB = True
