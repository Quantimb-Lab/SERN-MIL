import pytest
import torch
from torch import nn

from sern_mil.models.nfi.aggregation import NucleiFeatureAggregator
from sern_mil.models.nfi.student import NucleiStudent
from sern_mil.models.ser.decoders import assemble_patch_logits
from sern_mil.models.ser.module import SERModule
from sern_mil.models.ser.perceiver import PerceiverBottleneck
from sern_mil.models.ser.retrieval import select_informative_tokens
from sern_mil.training.losses import SERLoss


def test_assignment_favors_similarity_and_normalizes_over_prototypes():
    model = NucleiFeatureAggregator(2, 4, 2, 2, alpha=2)
    with torch.no_grad():
        model.prototypes.copy_(torch.eye(2))
    features = torch.tensor([[2.0, 0.0], [0.0, 1.0]], requires_grad=True)
    weights = model.assignments(features)
    assert torch.allclose(weights.sum(-1), torch.ones(2))
    assert weights[0, 0] > weights[0, 1]
    assert weights[1, 1] > weights[1, 0]
    model.phi, model.rho = nn.Identity(), nn.Identity()
    assert torch.allclose(model(features), (weights.T @ features).flatten())
    model(features).sum().backward()
    assert torch.isfinite(features.grad).all()
    assert torch.isfinite(model.prototypes.grad).all()


def test_nfi_permutation_padding_and_empty_nuclei():
    model = NucleiFeatureAggregator(2, 8, 2, 4).eval()
    features = torch.randn(1, 3, 2)
    actual = model(features)
    assert torch.allclose(actual, model(features[:, [2, 0, 1]]), atol=1e-6)
    padded = torch.cat([features, torch.randn(1, 4, 2)], dim=1)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    assert torch.allclose(actual, model(padded, mask), atol=1e-6)
    assert torch.equal(model(torch.empty(1, 0, 2)), torch.zeros(1, 8))


def test_student_contract():
    student = NucleiStudent()
    assert student(torch.randn(3, 1024), torch.zeros(3, 16, 16)).shape == (3, 128)
    with pytest.raises(ValueError):
        student(torch.randn(3, 1024), torch.zeros(3, 8, 8))


def test_perceiver_preserves_positions_and_uses_cross_patch_context():
    model = PerceiverBottleneck(8, 8, 2, 1, 2, 2, 0).eval()
    tokens = torch.randn(1, 4, 8)
    output = model(tokens)
    assert not torch.allclose(output[:, 0], output[:, 1])
    changed = tokens.clone()
    changed[:, 0] += torch.arange(8)
    assert not torch.allclose(output[:, 1], model(changed)[:, 1])
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 8))


def test_patch_placement_and_full_resolution_shape():
    patches = torch.arange(4).reshape(1, 4, 1, 1, 1).expand(-1, -1, -1, 2, 2)
    actual = assemble_patch_logits(patches, 2)
    assert actual[0, 0].tolist() == [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]
    patches = (
        torch.arange(256, dtype=torch.int16).reshape(1, 256, 1, 1, 1).expand(-1, -1, -1, 256, 256)
    )
    full = assemble_patch_logits(patches, 16)
    assert full.shape == (1, 1, 4096, 4096)
    assert full[0, 0, 0, 256] == 1
    assert full[0, 0, 256, 0] == 16
    assert full[0, 0, -1, -1] == 255


def test_ser_forward_backward_independent_decoders():
    model = SERModule(8, 8, 2, 1, 2, 2, 16, 4, 0, 2)
    tokens, shallow = torch.randn(1, 4, 8), torch.randn(1, 4, 4, 16, 16)
    output = model(tokens, shallow)
    assert output["tumor_logits"].shape == (1, 1, 32, 32)
    assert output["boundary_logits"].shape == (1, 1, 32, 32)
    loss = SERLoss()(
        output["tumor_logits"],
        output["boundary_logits"],
        torch.ones(1, 1, 32, 32),
        torch.zeros(1, 1, 32, 32),
    )
    loss.backward()
    assert model.bottleneck.encode_attn.in_proj_weight.grad is not None
    assert (
        model.tumor_head.project.weight.data_ptr() != model.boundary_head.project.weight.data_ptr()
    )


def test_retrieval_uses_tumor_area_not_boundary_and_no_fake_positive():
    mask = torch.full((1, 1, 20, 20), -10.0)
    mask[0, 0, :3, 10:] = 10
    mask[0, 0, 10:, 10:] = 10
    mask[0, 0, :2, :10] = 10
    tokens = torch.randn(1, 4, 8)
    selected, idx, occupancy = select_informative_tokens(
        tokens, mask, torch.full_like(mask, 10), patch_size=10, return_indices=True
    )
    assert idx[0].tolist() == [1, 3]
    assert selected[0].shape == (2, 8)
    assert occupancy[0, 1].item() == pytest.approx(0.3)
    empty = select_informative_tokens(tokens, torch.full_like(mask, -10), patch_size=10)
    assert empty[0].shape == (0, 8)


def test_boundary_loss_weight_schedule():
    assert SERLoss.boundary_beta(0) == 0.3
    assert SERLoss.boundary_beta(15) == pytest.approx(1.15)
    assert SERLoss.boundary_beta(30) == 2
    assert SERLoss.boundary_beta(39) == 2
