"""
Switch network is meant to have multiplication layers
"""

import torch
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
        in_segments: int = None,
        out_segments: int = None,
        normalization: Callable[[Any], Any] = None,
        resnet: bool = False,
        num_input_layers: int = 2,
    ) -> None:

        self._layers = [
            high_order_fc_layers(
                layer_type=layer_type,
                n=n,
                in_features=in_width,
                out_features=out_width,
                segments=in_segments,
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


class HighOrderSwitchLayer(Module):
    def __init__(
        self,
        layer_type: str,
        n: str,
        in_width: int,
        out_width: int,
        hidden_layers: int,
        hidden_width: int,
        scale: float = 2.0,
        n_in: int = None,
        n_out: int = None,
        n_hidden: int = None,
        rescale_output: bool = False,
        periodicity: float = None,
        non_linearity: Callable[[Tensor], Tensor] = None,
        in_segments: int = None,
        out_segments: int = None,
        hidden_segments: int = None,
        normalization: Callable[[Any], Any] = None,
        resnet: bool = False,
    ) -> None:
        """
        Args :
            layer_type: Type of layer
                "continuous", "discontinuous",
                "polynomial", "fourier",
                "product", "continuous_prod",
                "discontinuous_prod"
            n:  Base number of nodes (or fourier components).  If none of the others are set
                then this value is used.
            in_width: Input width.
            out_width: Output width
            hidden_layers: Number of hidden layers.
            hidden_width: Number of hidden units
            scale: Scale of the segments.  A value of 2 would be length 2 (or period 2)
            n_in: Number of input nodes for interpolation or fourier components.
            n_out: Number of output nodes for interpolation or fourier components.
            n_hidden: Number of hidden nodes for interpolation or fourier components.
            rescale_output: Whether to average the outputs
            periodicity: Whether to make polynomials periodic after given length.
            non_linearity: Whether to apply a nonlinearity after each layer (except output)
            in_segments: Number of input segments for each link.
            out_segments: Number of output segments for each link.
            hidden_segments: Number of hidden segments for each link.
            normalization: Normalization to apply after each layer (before any additional nonlinearity).
            resnet: True if layer output should be added to the previous.
        """
        super().__init__()
        layer_list = []
        n_in = n_in or n
        n_hidden = n_hidden or n
        n_out = n_out or n

        input_layer = high_order_fc_layers(
            layer_type=layer_type,
            n=n_in,
            in_features=in_width,
            out_features=hidden_width,
            segments=in_segments,
            rescale_output=rescale_output,
            scale=scale,
            periodicity=periodicity,
        )
        layer_list.append(input_layer)
        for i in range(hidden_layers):
            if normalization is not None:
                layer_list.append(normalization())
            if non_linearity is not None:
                layer_list.append(non_linearity)

            hidden_layer = high_order_fc_layers(
                layer_type=layer_type,
                n=n_hidden,
                in_features=hidden_width,
                out_features=hidden_width,
                segments=hidden_segments,
                rescale_output=rescale_output,
                scale=scale,
                periodicity=periodicity,
            )

            # This will add the result of the previous layer after normalization
            if resnet is True and i > 0:
                hidden_layer = SumLayer(layer_list=[hidden_layer, layer_list[-1]])
            layer_list.append(hidden_layer)

        if normalization is not None:
            layer_list.append(normalization())
        if non_linearity is not None:
            layer_list.append(non_linearity)

        output_layer = high_order_fc_layers(
            layer_type=layer_type,
            n=n_out,
            in_features=hidden_width,
            out_features=out_width,
            segments=out_segments,
            rescale_output=rescale_output,
            scale=scale,
            periodicity=periodicity,
        )
        layer_list.append(output_layer)
        self.model = nn.Sequential(*layer_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
