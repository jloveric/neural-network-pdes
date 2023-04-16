"""
Switch network is meant to have multiplication layers
"""

from torch.nn import Module
from typing import Callable, Any
from torch import Tensor
from high_order_layers_torch.layers import high_order_fc_layers


class SwitchLayer(Module):
    """
    Switch layer just creates 2 (or more) identical input layers
    and then multiplies the output of all those layers.  In effect
    one of the layers can turn of features of the other.
    """

    def __init__(
        self,
        layer_type: str,
        n: str,
        in_width: int,
        out_width: int,
        scale: float = 2.0,
        segments: int = None,
        normalization: Callable[[Any], Any] = None,
        num_input_layers: int = 2,
    ) -> None:

        self._layers = [
            high_order_fc_layers(
                layer_type=layer_type,
                n=n,
                in_features=in_width,
                out_features=out_width,
                segments=segments,
                rescale_output=False,
                scale=scale,
                periodicity=False,
            )
            for _ in range(num_input_layers)
        ]

        self._normalization = normalization

    def forward(self, x) -> Tensor:
        outputs = [layer(x) for layer in self._layers]

        final = outputs[0]
        for i in range(1, len(outputs)):
            final *= outputs[i]

        if self._normalization is not None:
            final = self._normalization(x)

        return final
