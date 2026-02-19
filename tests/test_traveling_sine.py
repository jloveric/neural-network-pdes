import torch
from neural_network_pdes.traveling_sine_network import TravelingSineMLP


def test_traveling_sine_mlp_forward_extracts_t_and_uses_it():
    model = TravelingSineMLP(
        in_width=2,
        out_width=3,
        hidden_width=8,
        hidden_layers=3,
        t_index=1,
        use_t_in_input=True,
        w_init=1.0,
        a_init=1.0,
    )

    x = torch.randn(16, 2, requires_grad=True)
    y = model(x)
    assert y.shape == (16, 3)

    loss = y.sum()
    loss.backward()
    assert x.grad is not None


def test_traveling_sine_mlp_forward_explicit_t():
    model = TravelingSineMLP(
        in_width=2,
        out_width=3,
        hidden_width=8,
        hidden_layers=2,
        t_index=1,
        use_t_in_input=False,
    )

    x = torch.randn(10, 2, requires_grad=True)
    t = torch.randn(10)
    y = model(x, t=t)
    assert y.shape == (10, 3)

    y.mean().backward()
    assert x.grad is not None
