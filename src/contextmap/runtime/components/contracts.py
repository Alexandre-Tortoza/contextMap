"""Nomes dos contratos que atravessam as arestas do DAG do runtime.

Cada nome identifica um tipo de artifact imutável de ``docs/PIPELINE.md``: é a string que
uma ``StageInput`` declara consumir e que uma ``StageDeclaration`` declara produzir. Este
módulo é uma folha de propósito -- não importa nada --, para que tanto os presets quanto os
executores e a fundação espacial falem o mesmo vocabulário sem depender da topologia nem do
registro de backends.

Os valores são persistidos no plano e no journal de cada run; mudar um deles muda a
identidade do que já foi produzido.
"""

SEQUENCE = "SequenceArtifact"
REGIONS = "RegionPerceptionRunArtifact"
VISUAL_EVIDENCE = "VisualEvidenceArtifact"
SEMANTIC_INTERPRETATION = "SemanticInterpretationArtifact"
SEMANTIC_SCORES = "SemanticScoringRunArtifact"
TRAJECTORY = "StateEstimationRunArtifact"
GEOMETRY = "GeometricMapArtifact"
ASSOCIATION = "SensorAssociationRunArtifact"
REPRESENTATION = "PointRepresentationRunArtifact"
FUSION = "SemanticFusionRunArtifact"
ENTITIES = "SemanticEntityArtifact"
RESOLUTION = "EntityResolutionRunArtifact"
RELATIONS = "SpatialRelationsRunArtifact"
CONTEXT_MAP = "ContextMapArtifact"
