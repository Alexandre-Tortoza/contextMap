"""The effective identity of the canonical discovery path (#636).

Changing a default of the pass plan or of the normalization, or one of the versioned algorithms
that turn proposals into canonical regions, used to leave every identity of the run unchanged.
"""

import dataclasses
import json

import pytest
from config_defaults import replace_default

from contextmap.region_perception import (
    BorderPolicy,
    DiscoveryPassConfig,
    NormalizationConfig,
    RegionDiscoveryIdentity,
    TilingConfig,
)
from contextmap.region_perception.discovery import region_discovery as discovery_module

_ALGORITHM_IDS = (
    "PASS_PLANNING_POLICY_ID",
    "BORDER_TRUNCATION_POLICY_ID",
    "MASK_REMAP_POLICY_ID",
)


def test_the_identity_names_the_current_algorithms() -> None:
    identity = RegionDiscoveryIdentity.effective()

    assert identity.mask_remap_policy_id == "nearest_pixel_center_v1"
    assert identity.pass_planning_policy_id == "row_major_edge_aligned_v1"
    assert identity.border_truncation_policy_id == "internal_edge_contact_v1"
    assert identity.pass_config_digest == DiscoveryPassConfig().digest
    assert identity.normalization_config_digest == NormalizationConfig().digest


def test_the_same_configuration_always_has_the_same_identity() -> None:
    passes = DiscoveryPassConfig(tiling=TilingConfig(tile_width=4, tile_height=4, overlap_x=1))
    normalization = NormalizationConfig(minimum_area_pixels=2.0)

    first = RegionDiscoveryIdentity.effective(passes, normalization)
    second = RegionDiscoveryIdentity.effective(passes, normalization)

    assert first == second
    assert first.digest == second.digest


def test_a_parameter_set_explicitly_to_its_default_does_not_change_the_identity() -> None:
    implicit = RegionDiscoveryIdentity.effective()

    explicit = RegionDiscoveryIdentity.effective(
        DiscoveryPassConfig(include_full_frame=True, max_candidates_per_pass=None),
        NormalizationConfig(duplicate_iou_threshold=0.8),
    )

    assert explicit == implicit
    assert explicit.digest == implicit.digest


def test_a_changed_normalization_default_changes_the_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = RegionDiscoveryIdentity.effective()

    replace_default(monkeypatch, NormalizationConfig, "duplicate_iou_threshold", 0.5)

    assert RegionDiscoveryIdentity.effective().digest != before.digest


def test_a_changed_pass_default_changes_the_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    before = RegionDiscoveryIdentity.effective()

    replace_default(monkeypatch, DiscoveryPassConfig, "max_candidates_per_pass", 5)

    assert RegionDiscoveryIdentity.effective().digest != before.digest


@pytest.mark.parametrize("algorithm", _ALGORITHM_IDS)
def test_a_new_version_of_a_discovery_algorithm_changes_the_identity(
    monkeypatch: pytest.MonkeyPatch, algorithm: str
) -> None:
    before = RegionDiscoveryIdentity.effective()

    monkeypatch.setattr(discovery_module, algorithm, "a_different_algorithm_v0")

    assert RegionDiscoveryIdentity.effective().digest != before.digest


def test_the_pass_digest_follows_the_grids_the_planner_walks() -> None:
    grid = TilingConfig(tile_width=4, tile_height=4, overlap_x=1, overlap_y=1)

    # As duas grafias planejam exatamente os mesmos passes, com os mesmos ids.
    assert (
        DiscoveryPassConfig(tiling=grid).digest
        == DiscoveryPassConfig(additional_tilings=(grid,)).digest
    )
    assert (
        DiscoveryPassConfig(tiling=grid).digest
        != DiscoveryPassConfig(
            tiling=dataclasses.replace(grid, border_policy=BorderPolicy.REJECT_INTERNAL_BORDER)
        ).digest
    )
    assert (
        DiscoveryPassConfig(tiling=grid).digest
        != DiscoveryPassConfig(tiling=dataclasses.replace(grid, scale=0.5)).digest
    )
    assert DiscoveryPassConfig(tiling=grid).digest != DiscoveryPassConfig().digest


def test_the_identity_round_trips_through_json() -> None:
    identity = RegionDiscoveryIdentity.effective(
        DiscoveryPassConfig(max_candidates_per_pass=3), NormalizationConfig(maximum_regions=7)
    )

    record = json.loads(json.dumps(identity.to_dict()))

    assert RegionDiscoveryIdentity.from_dict(record) == identity


def test_an_identity_that_does_not_name_an_algorithm_is_refused() -> None:
    record = RegionDiscoveryIdentity.effective().to_dict()
    record["mask_remap_policy_id"] = ""

    with pytest.raises(ValueError, match="mask_remap_policy_id"):
        RegionDiscoveryIdentity.from_dict(record)
