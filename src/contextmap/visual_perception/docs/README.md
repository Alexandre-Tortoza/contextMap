# Visual Perception

`visual_perception` possui features visuais, espaços de embedding, preparação de imagem,
persistência numérica e diagnósticos. Seu produto é `VisualEvidenceArtifact`.

A cadeia de percepção materializada usa três produtores independentes:

```text
SequenceArtifact → region_perception → RegionPerceptionRunArtifact
SequenceArtifact + RegionPerceptionRunArtifact → visual_perception → VisualEvidenceArtifact
SequenceArtifact + RegionPerceptionRunArtifact → semantic_interpretation → SemanticInterpretationArtifact
```

Regiões e máscaras pertencem a [Region Perception](../../region_perception/docs/README.md).
Claims, contexto, prompts e respostas pertencem a
[Semantic Interpretation](../../semantic_interpretation/docs/README.md).
Comparações entre features e hipóteses pertencem a
[Semantic Scoring](../../semantic_scoring/docs/README.md).

A API pública inclui `VisualFeature`, `FeatureId`, `FeatureScope`, `EmbeddingSpace`,
`FeatureExtractor`, `PreparedImage`, `VisualEvidenceResult`, `VisualEvidenceRunManifest`,
writer e reader visuais. `PerceptionResult` é somente uma view derivada de produtores
explicitamente selecionados; `PerceptionEvidenceSelection` e `PerceptionEvidenceSet`
validam e agrupam essas seleções sem fusão.

## Documentação

- [Contratos](contracts.md), [identidade](identity.md) e [ports](ports.md).
- [Artifact visual](visual-evidence-artifact.md), [stores](feature_store.md) e [diagnósticos](feature_diagnostics.md).
- [Seleção de evidência](evidence_set.md) e [execução](pipeline.md).
- [Arquitetura](architecture.md), [fluxo](data-flow.md) e [mapa do código](code-map.md).
- [Features visuais](../visual_features/docs/README.md), [testes](testing.md) e [invariantes científicos](scientific-invariants.md).

Backends concretos são compostos pelo runtime; nenhuma capability carrega configurações
de runtime nem recebe SDKs ou tensores nativos como contratos públicos.
