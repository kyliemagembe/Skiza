import torch

from skiza import CNN, SingleChannelCNN, convolve, mimo_convolve, mimo_convolve_fast, shift


def test_shift_delays_and_zero_fills():
    x = torch.arange(1.0, 6.0)
    assert torch.equal(shift(x, 2), torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0]))


def test_siso_convolution_matches_numpy():
    torch.manual_seed(0)
    x, h = torch.randn(50), torch.randn(7)
    expected = torch.from_numpy(__import__("numpy").convolve(x.numpy(), h.numpy())[:50])
    assert torch.allclose(convolve(x, h), expected, atol=1e-5)


def test_fast_mimo_matches_reference():
    torch.manual_seed(0)
    X = torch.randn(3, 4, 200)
    H = torch.randn(9, 4, 6)
    assert torch.allclose(mimo_convolve(X, H), mimo_convolve_fast(X, H), atol=1e-4)


def test_cnn_fast_and_reference_agree():
    torch.manual_seed(0)
    model = CNN([40, 40], [1, 5, 1], fast=True)
    ref = CNN([40, 40], [1, 5, 1], fast=False)
    ref.load_state_dict(model.state_dict())
    x = torch.randn(2, 1, 500) * 0.3
    assert torch.allclose(model(x), ref(x), atol=1e-5)


def test_cnn_is_shift_equivariant():
    """Lab Proposition 1: Phi(Sx) = S Phi(x) (up to the zero-filled edge)."""
    torch.manual_seed(0)
    model = CNN([10, 10], [1, 5, 1])
    x = torch.randn(1, 1, 300) * 0.3
    d = 17
    with torch.no_grad():
        assert torch.allclose(model(shift(x, d)), shift(model(x), d), atol=1e-6)


def test_cnn_is_causal():
    torch.manual_seed(0)
    model = CNN([40, 40], [1, 5, 1])
    x = torch.randn(1, 1, 400) * 0.3
    x2 = x.clone()
    x2[..., 200:] += 1.0  # change only the future
    with torch.no_grad():
        assert torch.allclose(model(x)[..., :200], model(x2)[..., :200])


def test_shapes():
    assert SingleChannelCNN([40, 40])(torch.zeros(4, 1, 100)).shape == (4, 1, 100)
    assert CNN([40, 40], [1, 5, 1])(torch.zeros(4, 100)).shape == (4, 100)
