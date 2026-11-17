# Semantic Scoring possui owner próprio

Scorers e torres de texto pertencem a `contextmap.semantic_scoring`. O stage seleciona
separadamente regiões, features visuais e interpretações semânticas, validando a lineage
antes da comparação. Scores não são probabilidades implícitas e não modificam claims.

Veja a [API e arquitetura atuais](../../semantic_scoring/docs/README.md) e o
[protocolo científico CLIP/AlphaCLIP](../../semantic_scoring/docs/scoring-protocol.md).
As features e suas fábricas de embedding space continuam pertencendo ao domínio visual.
