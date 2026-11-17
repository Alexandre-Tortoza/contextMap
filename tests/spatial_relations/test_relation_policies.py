"""The effective policies of a run: coherent among themselves, or refused at construction."""

from __future__ import annotations

import dataclasses
import warnings

import pytest
from relation_run_fixture import CANDIDATES, CONTACT, GEOMETRIC, POLICIES

from contextmap.spatial_relations.policies import RelationsRunPolicies


def test_the_policy_set_aggregates_every_declared_channel() -> None:
    assert isinstance(POLICIES, RelationsRunPolicies)
    assert POLICIES.geometric is GEOMETRIC
    assert POLICIES.contact is CONTACT
    assert POLICIES.candidate is CANDIDATES
    # O canal semântico é opcional e esta cena não o declara.
    assert POLICIES.semantic is None


def test_a_next_to_gap_beyond_the_proximity_reach_is_refused() -> None:
    wider = dataclasses.replace(GEOMETRIC, next_to_max_gap_m=0.9)
    with pytest.raises(ValueError) as error:
        dataclasses.replace(POLICIES, geometric=wider)
    message = str(error.value)
    assert "next_to_max_gap_m=0.9 exceeds proximity_radius_m=0.6" in message
    assert "NEXT_TO relations would be excluded" in message
    assert "before evaluation" in message


def test_a_contact_reach_beyond_the_proximity_reach_is_refused() -> None:
    # Cada parâmetro cabe sozinho no alcance; é a soma, o raio de busca real do canal de contato,
    # que passa dele.
    farther = dataclasses.replace(CONTACT, contact_distance_m=0.5, contact_tolerance_m=0.2)
    with pytest.raises(ValueError) as error:
        dataclasses.replace(POLICIES, contact=farther)
    message = str(error.value)
    assert "contact_distance_m + contact_tolerance_m = 0.7 (0.5 + 0.2)" in message
    assert "exceeds proximity_radius_m=0.6" in message
    assert "TOUCHING, ON_TOP_OF and LEANING_AGAINST relations would be excluded" in message


def test_a_containment_slack_on_both_faces_beyond_the_proximity_reach_is_refused() -> None:
    # containment_slack_m <= proximity_radius_m < 2 * containment_slack_m: o sujeito pode
    # ultrapassar as duas faces de um eixo, então o avaliador aceita um excesso de extensão
    # que a pré-condição de INSIDE já teria descartado.
    looser = dataclasses.replace(GEOMETRIC, containment_slack_m=0.4)
    with pytest.raises(ValueError) as error:
        dataclasses.replace(POLICIES, geometric=looser)
    message = str(error.value)
    assert "2 * containment_slack_m = 0.8" in message
    assert "containment_slack_m=0.4" in message
    assert "exceeds proximity_radius_m=0.6" in message
    assert "INSIDE relations would be excluded" in message


def test_every_incoherence_is_reported_at_once() -> None:
    short = dataclasses.replace(CANDIDATES, proximity_radius_m=0.04)
    with pytest.raises(ValueError) as error:
        dataclasses.replace(POLICIES, candidate=short)
    message = str(error.value)
    for name in ("next_to_max_gap_m=", "2 * containment_slack_m", "contact_distance_m +"):
        assert name in message


def test_policies_whose_reach_covers_every_tolerance_are_accepted_silently() -> None:
    # Os limites são inclusivos: um alcance exatamente igual à tolerância não perde nada.
    geometric = dataclasses.replace(GEOMETRIC, next_to_max_gap_m=0.6, containment_slack_m=0.3)
    contact = dataclasses.replace(CONTACT, contact_distance_m=0.5, contact_tolerance_m=0.1)
    short = dataclasses.replace(CANDIDATES, proximity_radius_m=0.04)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        at_the_limit = dataclasses.replace(POLICIES, geometric=geometric, contact=contact)
        # Um canal ausente não tem tolerância a cobrir, por menor que seja o alcance.
        without_channels = dataclasses.replace(
            POLICIES, candidate=short, geometric=None, contact=None
        )
    assert at_the_limit.geometric == geometric
    assert at_the_limit.contact == contact
    assert without_channels.candidate == short
