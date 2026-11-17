# Semantic Interpretation possui owner próprio

Requests, prompts, views exatas, parsing, claims e contexto de cena pertencem a
`contextmap.semantic_interpretation`, que publica `SemanticInterpretationArtifact`
independentemente das features visuais. `PerceptionResult` é somente uma view derivada
de produtores selecionados; não é a unidade persistida da interpretação.

Veja a [API e arquitetura atuais](../../semantic_interpretation/docs/README.md),
o [protocolo científico](../../semantic_interpretation/docs/semantic-interpretation.md) e
o [registro histórico preservado](../../semantic_interpretation/docs/history/pre-decomposition-interpretation.md).
