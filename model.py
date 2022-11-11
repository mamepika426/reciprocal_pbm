from dataclasses import dataclass
from typing import Tuple

from torch import FloatTensor, nn


# dataclassについて: https://hombre-nuevo.com/python/python0083/
@dataclass(unsafe_hash=True)
class MLPScoreFunc(nn.Module):
    """多層パーセプトロンによるスコアリング関数.
    @param input_size: 特徴量ベクトルの次元数.
    @param hidden_layer_sizes: 隠れ層におけるニューロン数を定義するタプル
    @param activation_func: 活性化関数.
    """
    input_size: int
    hidden_layer_sizes: Tuple[int, ...]
    activation_func: nn.functional = nn.functional.elu

    def __post_init__(self) -> None:
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(
            nn.Linear(self.input_size, self.hidden_layer_sizes[0])
        )
        for hin, hout in zip(self.hidden_layer_sizes, self.hidden_layer_sizes[1:]):
            self.hidden_layers.append(nn.Linear(hin, hout))
        self.output = nn.Linear(self.hidden_layer_sizes[-1], 1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        h = x
        for layer in self.hidden_layers:
            h = self.activation_func(layer(h))
        return self.output(h).flatten(1)
