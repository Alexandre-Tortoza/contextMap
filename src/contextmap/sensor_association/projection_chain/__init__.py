"""A cadeia determinística que leva a geometria do mapa até um pixel da imagem preparada.

Para uma observação RGB, os quatro passos deste subpacote decidem **onde** cada elemento do
mapa cai e **se** ele estava visível:

```text
seleção de candidatos -> P_map -> P_camera em t_rgb -> pixel bruto -> pixel preparado
    -> visibilidade por suporte local de profundidade
```

`candidate_geometry` escolhe, antes de qualquer projeção exata, que geometria o frame avalia;
`image_transform` reproduz a cadeia de preparo da imagem registrada por Visual Perception;
`frame_projection` compõe pose, extrínseca e modelo de câmera e coloca cada ponto no espaço
preparado; `visibility` decide quem está ocluído.

Nada aqui escolhe região, atribui rótulo ou produz contrato público: um ponto que chega
visível ao fim da cadeia ainda é só um candidato para o pertencimento.
"""

from contextmap.sensor_association.projection_chain.candidate_geometry import (
    CANDIDATE_GEOMETRY_POLICY_ID,
    CandidateGeometryCloud,
    CandidateGeometryPolicy,
    CandidateGeometryReport,
    CandidateGeometrySelection,
    locate_global_indices,
    select_candidate_geometry,
)
from contextmap.sensor_association.projection_chain.frame_projection import (
    AuditedProjection,
    ExtrinsicRef,
    FrameProjection,
    FrameProjector,
    ProjectionStage,
    RejectedProjection,
)
from contextmap.sensor_association.projection_chain.image_transform import (
    ImageTransformStep,
    RawToPreparedTransform,
    raw_to_prepared_transform,
)
from contextmap.sensor_association.projection_chain.visibility import (
    OcclusionPolicy,
    VisibilityResolution,
    depth_metric_for,
    resolve_visibility,
)

__all__ = [
    "CANDIDATE_GEOMETRY_POLICY_ID",
    "AuditedProjection",
    "CandidateGeometryCloud",
    "CandidateGeometryPolicy",
    "CandidateGeometryReport",
    "CandidateGeometrySelection",
    "ExtrinsicRef",
    "FrameProjection",
    "FrameProjector",
    "ImageTransformStep",
    "OcclusionPolicy",
    "ProjectionStage",
    "RawToPreparedTransform",
    "RejectedProjection",
    "VisibilityResolution",
    "depth_metric_for",
    "locate_global_indices",
    "raw_to_prepared_transform",
    "resolve_visibility",
    "select_candidate_geometry",
]
