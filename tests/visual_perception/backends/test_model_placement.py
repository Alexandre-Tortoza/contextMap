"""Model placement is verified over every tensor of the model, not a sample (#646)."""

import pytest
from fakes import FakeLoadedModel, fake_tensor

from contextmap.model_runtime import verify_model_placement


def test_a_homogeneous_model_placed_as_configured_is_accepted() -> None:
    model = FakeLoadedModel(
        "cuda:0",
        "float32",
        others=(fake_tensor("cuda:0", "float32"),),
        buffers=(fake_tensor("cuda:0", "float32"),),
    )

    dtype = verify_model_placement(model, device="cuda", precision="float32", backend="SAM2")

    assert dtype == "torch.float32"


def test_a_parameter_after_the_first_on_another_device_is_refused() -> None:
    model = FakeLoadedModel(
        "cuda:0",
        "float32",
        others=(fake_tensor("cuda:0", "float32"), fake_tensor("cpu", "float32")),
    )

    with pytest.raises(ValueError, match=r"cpu \(1 tensor\), cuda:0 \(2 tensors\)"):
        verify_model_placement(model, device="cuda:0", precision="float32", backend="SAM2")


def test_a_model_spread_over_two_devices_of_the_configured_type_is_refused() -> None:
    # Placement distribuído (device_map/offload) nunca é aceito em silêncio: sem índice na
    # configuração, cada tensor sozinho conferiria, mas o conjunto não é um único device.
    model = FakeLoadedModel("cuda:0", "float32", others=(fake_tensor("cuda:1", "float32"),))

    with pytest.raises(ValueError, match="distributed placement"):
        verify_model_placement(model, device="cuda", precision="float32", backend="SAM2")


def test_a_buffer_left_on_another_device_is_refused() -> None:
    model = FakeLoadedModel("cuda:0", "float32", buffers=(fake_tensor("cpu", "int64"),))

    with pytest.raises(ValueError, match=r"cpu \(1 tensor\)"):
        verify_model_placement(model, device="cuda:0", precision="float32", backend="SAM2")


def test_the_dtype_of_a_buffer_is_not_a_weight_precision() -> None:
    # Buffers guardam estado que não é peso (ids de posição inteiros, por exemplo).
    model = FakeLoadedModel("cuda:0", "float32", buffers=(fake_tensor("cuda:0", "int64"),))

    verify_model_placement(model, device="cuda:0", precision="float32", backend="SAM2")


def test_heterogeneous_weights_outside_the_rule_are_refused() -> None:
    model = FakeLoadedModel("cuda:0", "float32", others=(fake_tensor("cuda:0", "float16"),))

    with pytest.raises(ValueError, match=r"torch.float16 \(1 tensor\), torch.float32 \(1 tensor\)"):
        verify_model_placement(model, device="cuda:0", precision="float32", backend="SAM2")


def test_heterogeneous_weights_inside_the_rule_are_accepted() -> None:
    # Regra do SAM3 (#617): sob autocast bfloat16, pesos float32 e bfloat16 inferem em bfloat16.
    model = FakeLoadedModel("cuda:0", "float32", others=(fake_tensor("cuda:0", "bfloat16"),))

    verify_model_placement(
        model,
        device="cuda:0",
        precision="bfloat16",
        backend="SAM3",
        accepted_dtypes=frozenset({"float32", "bfloat16"}),
    )


def test_a_model_without_parameters_is_refused() -> None:
    class Empty:
        def parameters(self) -> list[object]:
            return []

    with pytest.raises(ValueError, match="no parameter"):
        verify_model_placement(Empty(), device="cpu", precision="float32", backend="SAM2")
