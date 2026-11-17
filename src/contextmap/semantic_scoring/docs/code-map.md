# Mapa do código de Semantic Scoring

| Arquivo | Responsabilidade |
| --- | --- |
| `contracts.py` | `ScoreId`, `SemanticScoreType`, `SemanticScore`; valida identidade, valor finito, faixa e proveniência |
| `ports.py` | `SemanticScorer`: proveniência e scoring em lote |
| `artifacts.py` | Linhagem, manifest, codecs, validação da view e publicação/leitura do artifact |
| `__init__.py` | API pública explícita de contratos, port e persistência |
| `adapters/_scoring_base.py` | Pareamento compatível, template, payload e regra compartilhada de cosseno |
| `adapters/clip.py` | `ClipSemanticScorer` e `ClipRegionSemanticScorer` |
| `adapters/alphaclip.py` | `AlphaClipSemanticScorer` |
| `adapters/clip_text.py` | Torre de texto CLIP para hipóteses do scorer |
| `adapters/alphaclip_text.py` | Torre de texto AlphaCLIP para hipóteses do scorer |

## Contratos e persistência

`ClaimId` vem de semantic interpretation, `RegionId` de region perception e `FeatureId` de
visual perception. `SourceObservationId` vem de ingestion e `BackendProvenance` de shared.
Consumidores obtêm `SemanticScore` pela raiz pública desta capability.

`SemanticScoringRunWriter.add_result(view, scores)` recebe uma `PerceptionResult` derivada,
sem identidade de inferência própria, e valida os três produtores upstream declarados na
linhagem. A view reúne evidências para a operação; não republica os artifacts de entrada.

`finalize()` escreve `outputs/scores.jsonl`, agrega cobertura e publica atomicamente.
`read_manifest()` retorna tipos de domínio para espaço, escopo, semântica e inventário;
`read_scores()` retorna scores tipados; `iter_scores()` permite percorrer o stream;
`verify_integrity()` retorna divergências de hashes, tamanho ou arquivos ausentes.
O inventário publicado é injetado pelo publisher e restaurado pelo reader.

## Dependências

O núcleo não calcula projeção, interpreta imagens, extrai features ou seleciona backends.
Também não importa SDKs ou módulos internos de outra capability. As torres de texto e os
scorers concretos são conectados pelo composition root em runtime. O ciclo antigo com
extração visual não faz parte da arquitetura atual.

Detalhes: [arquitetura](architecture.md), [fluxo](data-flow.md) e [testes](testing.md).
