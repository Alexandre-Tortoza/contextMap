"""Agrupamento de conjuntos de geometria pelas suas sobreposições de Jaccard.

Mecanismo puro: recebe, para cada observação, o conjunto de elementos de geometria que
ela vê, e devolve as componentes conexas das ligações ``jaccard >= min_overlap``. Não
conhece observação espacial, política de suporte, limiar científico nem identidade de
suporte -- quem decide isso é ``support_construction``.

Este módulo existe separado porque muda por razão de **custo**, não de ciência. A
sobreposição par a par usa um bit set por observação sobre a geometria que ao menos
duas observações referenciam, e os pares candidatos vêm de um índice invertido dessa
geometria compartilhada: só observações que compartilham algum elemento são comparadas,
então o custo segue o tamanho da entrada mais os pares que realmente se intersectam, e é
quadrático apenas quando toda observação se sobrepõe a todas as outras -- onde esses
pares são trabalho real.

A componente é identificada pela menor posição que a compõe, então a ordem de saída
segue a primeira observação de cada componente.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence

from contextmap.geometric_mapping import GeometryId


def overlap_components(
    geometry_sets: Sequence[Sequence[GeometryId]], *, min_overlap: float
) -> list[list[int]]:
    """Link the given geometry sets by Jaccard overlap and return the connected components.

    Args:
        geometry_sets: For each observation, in input order, the identities of the geometry it
            sees. Each set must already be unique; the caller owns that invariant.
        min_overlap: The Jaccard overlap at which two observations are linked. The threshold
            itself links them.

    Returns:
        The components, each a list of input positions in increasing order, ordered by their
        first member. The linkage is transitive, so a chain of partly overlapping sets forms
        one component even when its ends do not overlap each other.
    """
    counts = [len(item) for item in geometry_sets]
    # Índice invertido: quem vê cada elemento (cada conjunto já é único por observação).
    holders: defaultdict[GeometryId, list[int]] = defaultdict(list)
    for index, item in enumerate(geometry_sets):
        for geometry_id in item:
            holders[geometry_id].append(index)
    # Só a geometria vista por mais de uma observação pode contribuir para uma interseção. Os
    # elementos vistos pelo mesmo conjunto de observações formam uma classe, e cada classe ocupa
    # uma faixa contígua de bits: o bit set de uma observação é o OR das faixas das suas classes,
    # e seus parceiros, o OR dos conjuntos de observações dessas classes. Um bit por elemento,
    # então a interseção de um par é exata.
    class_sizes: dict[tuple[int, ...], int] = {}
    for seen_by in holders.values():
        if len(seen_by) > 1:
            key = tuple(seen_by)
            class_sizes[key] = class_sizes.get(key, 0) + 1
    spans: list[list[tuple[int, int]]] = [[] for _ in geometry_sets]
    partners = [0] * len(geometry_sets)
    width = 0
    for members_of_class, size in class_sizes.items():
        class_bits = _index_bit_set(members_of_class)
        for index in members_of_class:
            spans[index].append((width, size))
            partners[index] |= class_bits
        width += size
    masks = [_span_bit_set(item, width) for item in spans]

    parent = list(range(len(geometry_sets)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(geometry_sets)):
        # Só os pares que compartilham geometria podem se ligar: nenhum par disjunto é avaliado.
        for right in _set_bits(partners[left] >> (left + 1), offset=left + 1):
            if (
                jaccard_overlap(masks[left], masks[right], counts[left], counts[right])
                >= min_overlap
            ):
                root_left, root_right = find(left), find(right)
                if root_left != root_right:
                    # A menor posição é a raiz: a primeira observação identifica a componente.
                    parent[max(root_left, root_right)] = min(root_left, root_right)

    members: dict[int, list[int]] = {}
    for index in range(len(geometry_sets)):
        members.setdefault(find(index), []).append(index)
    return [members[root] for root in sorted(members)]


def jaccard_overlap(left_mask: int, right_mask: int, left_count: int, right_count: int) -> float:
    """Jaccard index of two observations, from their shared-geometry bit sets and sizes.

    Args:
        left_mask: Bit set of the shared geometry the left observation sees.
        right_mask: The same for the right observation.
        left_count: Total geometry the left observation sees, shared or not.
        right_count: The same for the right observation.

    Returns:
        ``|A and B| / |A or B|``, counting only the identity of the geometry.
    """
    intersection = (left_mask & right_mask).bit_count()
    return intersection / (left_count + right_count - intersection)


def _index_bit_set(indexes: tuple[int, ...]) -> int:
    """Bit set with one bit per observation position in ``indexes``."""
    buffer = bytearray((max(indexes) >> 3) + 1)
    for index in indexes:
        buffer[index >> 3] |= 1 << (index & 7)
    return int.from_bytes(buffer, "little")


def _span_bit_set(spans: list[tuple[int, int]], width: int) -> int:
    """Bit set of ``width`` bits with ``[start, start + size)`` set for every span, in one pass."""
    buffer = bytearray((width + 7) >> 3)
    for start, size in spans:
        if size == 1:
            buffer[start >> 3] |= 1 << (start & 7)
            continue
        end = start + size
        # Bits soltos até o próximo byte inteiro, bytes inteiros no meio, bits soltos no fim.
        head = min(end, (start + 7) & ~7)
        tail = max(head, end & ~7)
        for bit in (*range(start, head), *range(tail, end)):
            buffer[bit >> 3] |= 1 << (bit & 7)
        buffer[head >> 3 : tail >> 3] = b"\xff" * ((tail - head) >> 3)
    return int.from_bytes(buffer, "little")


def _set_bits(bits: int, *, offset: int) -> Iterator[int]:
    """Yield ``offset`` plus the position of every set bit of ``bits``, in increasing order."""
    while bits:
        lowest = bits & -bits
        yield offset + lowest.bit_length() - 1
        bits ^= lowest
