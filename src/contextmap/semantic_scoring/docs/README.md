# Semantic Scoring

Esta capability julga a compatibilidade entre uma hipótese de `SemanticClaim` e uma
`VisualFeature`. Produz `SemanticScore` com semântica numérica explícita e um
`SemanticScoringRunArtifact`, schema `0.2.0`, separado das evidências de entrada.

## Ownership e entradas

| Conceito | Owner e API pública |
| --- | --- |
| Claim e `ClaimId` | `contextmap.semantic_interpretation` |
| Região e `RegionId` | `contextmap.region_perception` |
| Feature, `FeatureId` e espaço de embedding | `contextmap.visual_perception` |
| Score, scorer e artifact de scoring | `contextmap.semantic_scoring` |
| Proveniência de backend | `contextmap.shared` |

O port recebe lotes de claims e features. O stage de runtime seleciona três artifacts:
`regions` de `region_perception`, `visual_evidence` de `visual_perception` e
`semantic_interpretation` de `semantic_interpretation`. O stage é opcional e desabilitado
por padrão; produzir regiões, features ou claims não executa scoring automaticamente.

`PerceptionResult`, quando usado por `add_result`, é uma view derivada da mesma observação e
produtores selecionados. Seus campos de linhagem apontam para artifacts independentes;
a view não possui identidade de run ou resultado própria.

## Semântica e invariantes

- `COSINE_SIMILARITY` preserva `value` em `[-1, 1]`. Não é probabilidade.
- Claim sem feature compatível permanece sem score; nenhum placeholder zero é criado.
- O pareamento exige espaço de embedding, escopo e ancoragem compatíveis.
- Scores não modificam claims, features ou seus artifacts.
- Os adapters atuais não produzem `calibrated_probability`; o writer recusa esse campo
  preenchido, pois calibração externa exige contrato e proveniência próprios.
- Falhas não selecionam outro scorer silenciosamente.

`SemanticScore` registra `score_id`, `claim_id`, `feature_id`, `source_observation_id`,
`region_id`, `embedding_space_id`, tipo, valor e proveniência do scorer. O writer qualifica
`score_id` pelo run de scoring. A linhagem do manifest registra sequência, seleção e os
três produtores upstream.

## Persistência e consumo

O writer valida a view e os scores com `add_result(view, scores)` e publica com `finalize()`.
Ele recusa produtores divergentes, observação repetida, referências ausentes, pareamento de
região incompatível e espaço ou scorer diferentes. A publicação usa staging, inventário
SHA-256 e rename atômico; um diretório finalizado nunca é substituído.

```python
from contextmap.semantic_scoring import SemanticScoringRunReader

reader = SemanticScoringRunReader(artifact_dir)
manifest = reader.read_manifest()
scores = reader.read_scores()
problems = reader.verify_integrity()
```

Abrir o artifact de scoring não exige carregar os upstreams nem SDKs de modelo. Para validar
uma junção científica, o consumidor seleciona e verifica os artifacts upstream explícitos.

A API pública também expõe os contratos, `SemanticScorer`, writer, manifest, linhagem, erro
e codecs. Adapters concretos e torres de texto ficam internos; runtime os compõe.

- [Arquitetura e fronteiras](architecture.md)
- [Fluxo e persistência](data-flow.md)
- [Mapa do código](code-map.md)
- [Testes e avaliação](testing.md)
