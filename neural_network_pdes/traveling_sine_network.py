import torch
import torch.nn as nn
from torch import Tensor


class TravelingSineActivation(nn.Module):
    def __init__(self, width: int, w_init: float = 1.0, a_init: float = 1.0):
        super().__init__()
        self.w = nn.Parameter(torch.full((width,), float(w_init)))
        self.a = nn.Parameter(torch.full((width,), float(a_init)))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(1)
        return torch.sin(self.w * (x - self.a * t))


class TravelingSineMLP(nn.Module):
    def __init__(
        self,
        in_width: int,
        out_width: int,
        hidden_width: int,
        hidden_layers: int,
        t_index: int = -1,
        use_t_in_input: bool = True,
        w_init: float = 1.0,
        a_init: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")

        self.t_index = int(t_index)
        self.use_t_in_input = bool(use_t_in_input)

        linear_in_width = int(in_width) if self.use_t_in_input else int(in_width) - 1
        if linear_in_width < 1:
            raise ValueError("in_width is too small for use_t_in_input=False")

        self.in_linear = nn.Linear(linear_in_width, hidden_width, bias=bias)
        self.in_act = TravelingSineActivation(
            width=hidden_width, w_init=w_init, a_init=a_init
        )

        self.hidden_linears = nn.ModuleList(
            [nn.Linear(hidden_width, hidden_width, bias=bias) for _ in range(hidden_layers - 1)]
        )
        self.hidden_acts = nn.ModuleList(
            [
                TravelingSineActivation(width=hidden_width, w_init=w_init, a_init=a_init)
                for _ in range(hidden_layers - 1)
            ]
        )

        self.out_linear = nn.Linear(hidden_width, out_width, bias=bias)

    def _split_input(self, x: Tensor, t: Tensor | None) -> tuple[Tensor, Tensor]:
        if t is None:
            t = x[:, self.t_index]
        if t.ndim == 1:
            t = t.unsqueeze(1)

        if self.use_t_in_input:
            x_in = x
        else:
            t_col = self.t_index if self.t_index >= 0 else x.shape[1] + self.t_index
            x_in = torch.cat([x[:, :t_col], x[:, t_col + 1 :]], dim=1)

        return x_in, t

    def forward(self, x: Tensor, t: Tensor | None = None) -> Tensor:
        x_in, t_in = self._split_input(x, t)
        h = self.in_linear(x_in)
        h = self.in_act(h, t_in)
        for lin, act in zip(self.hidden_linears, self.hidden_acts):
            h = lin(h)
            h = act(h, t_in)
        return self.out_linear(h)
