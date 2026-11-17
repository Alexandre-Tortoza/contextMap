# Arquitetura visual

`visual_perception` possui features e sua persistência. Consome regiões finais pela API
pública de `region_perception`; contexto e claims pertencem a `semantic_interpretation`.
Os três stages têm configuração, execução, identidade e artifacts independentes.

O runtime seleciona os artifacts e constrói os adapters. O executor visual opera somente
extração densa, global ou regional e pooling opcional. Modelos recebem imagem preparada,
regiões congeladas e um resolver de máscaras do upstream regional; nunca alteram esse upstream.

A API pública expõe contratos e readers/writers necessários aos consumidores. Adapters,
SDKs e tensores nativos permanecem internos. As torres de texto pertencem a Semantic Scoring;
as fábricas puras de embedding space permanecem no domínio de features visuais.

`PerceptionEvidenceSet` compõe views dos produtores explicitamente selecionados. Essas
views preservam provenance, não persistem outro artifact e não executam fusão ou resolução.

Veja [arquitetura das features](../visual_features/docs/architecture.md) e
[arquitetura global](../../../../docs/architecture.md).
