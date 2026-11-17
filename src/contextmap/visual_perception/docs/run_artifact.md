# Artifacts de percepção

A percepção publica três artifacts independentes e imutáveis:

| Produtor | Artifact | Evidência que possui |
| --- | --- | --- |
| `region_perception` | `RegionPerceptionRunArtifact` | frames processados, regiões finais, máscaras, discovery, grounding e refinamento |
| `visual_perception` | `VisualEvidenceArtifact` | features, payloads e diagnósticos de extração |
| `semantic_interpretation` | `SemanticInterpretationArtifact` | requests, views exatas, respostas brutas, parsing, claims e contexto de cena |

Cada manifest identifica sequência, seleção, configuração, código e upstreams. Features e
claims referenciam as regiões do artifact regional selecionado; os consumidores validam
as identidades explícitas antes de combinar evidência.

`PerceptionResult` é uma view derivada para uma observação física. Ela reúne uma seleção
explícita desses produtores, com scores opcionais, sem `result_id`, `run_id` próprio ou
serialização como artifact monolítico. Construir a view não copia payloads nem executa fusão.

Os writers publicam atomicamente e recusam substituir artifacts existentes. Readers
abrem metadados sem carregar todos os payloads; máscaras e features são resolvidas sob
demanda com verificação de hash. Dados contratuais ficam em `outputs/` e `metrics/`;
`__DEBUG__/` permanece somente diagnóstico humano.

Veja [artifact visual](visual-evidence-artifact.md),
[artifact regional](../../../../docs/ARTIFACTS.md#regionperceptionrunartifact),
[artifact semântico](../../semantic_interpretation/docs/README.md) e
[seleção de evidência](evidence_set.md).
