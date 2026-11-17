# Fluxo de evidência visual

```text
SequenceArtifact + RegionPerceptionRunArtifact selecionado
    → PreparedImage consistente com o frame regional
    → extração de features e pooling opcional
    → metadata + payloads + diagnósticos em staging
    → VisualEvidenceArtifact imutável
```

A preparação conserva a geometria e elegibilidade usadas pelo produtor regional.
Máscaras são resolvidas no artifact regional; o artifact visual referencia o upstream.
Arrays são gravados imediatamente, validados e inventariados antes da publicação atômica.

Semantic Interpretation usa seu próprio stage sobre a sequência e o artifact regional.
Semantic Scoring seleciona separadamente os produtores regional, visual e semântico.
Sensor Association pode consumir features densas como entrada opcional; sua associação
geométrica depende do produtor regional, não de claims. Semantic Fusion seleciona os
produtores pertinentes e aplica a acumulação correlacionada por observação física.

Veja [fluxo das features](../visual_features/docs/data-flow.md) e
[pipeline global](../../../../docs/PIPELINE.md).
