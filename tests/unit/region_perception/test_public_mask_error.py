"""Consumers can report the mask owner's contractual error."""


def test_mask_store_error_is_owned_by_public_regional_api() -> None:
    from contextmap.region_perception import MaskStoreError
    from contextmap.region_perception.artifacts.mask_store import MaskStoreError as OwnedError

    assert MaskStoreError is OwnedError
