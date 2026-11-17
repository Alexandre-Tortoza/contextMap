# Arquitetura de Semantic Scoring

## Fronteiras

Scoring possui o julgamento de compatibilidade cross-modal. `semantic_interpretation`
possui claims e contexto; `region_perception` possui regiões e máscaras; `visual_perception`
possui features e seus payloads. `semantic_fusion` decide como acumular a evidência de scores.
Runtime seleciona artifacts e backends e coordena a execução.

O núcleo importa contratos pelas APIs públicas dos owners: `ClaimId` e `SemanticClaim`
semânticos, `RegionId` regional, `FeatureId` e `VisualFeature` visuais,
`SourceObservationId` de ingestion e `BackendProvenance` de shared. Não há ID de resultado
monolítico no contrato de score. SDKs, tensores nativos e torres de modelo ficam nos adapters.

`SemanticScorer` declara `backend_provenance()` e `score(claims, features)`. A interface em lote
permite filtrar os conjuntos juntos e codificar hipóteses sem repetição desnecessária.
Liberação de recursos pertence ao contrato de lifecycle usado por runtime; `release()` não
é requisito de um scorer puro.

## Seleção explícita e identidade

O stage `semantic_scoring` é opcional e desabilitado por padrão. Seus inputs são os artifacts
regional, visual e semântico. Eles compartilham observação física e produtor regional;
o executor verifica essa compatibilidade antes de montar a view de leitura.

`SemanticScoringLineage` guarda `sequence_artifact_id`, `selection_id`, `region_artifact_id`,
`visual_evidence_artifact_id` e `semantic_interpretation_artifact_id`. O writer recusa uma view
que declare outros produtores e aceita cada `source_observation_id` apenas uma vez.

`SemanticScore` referencia claim e feature por ID, observação e região. A identidade do run
que julgou o par fica no artifact de scoring e qualifica os scores publicados; ela não
substitui a identidade dos produtores das evidências.

## Adapters e semântica científica

`ClipSemanticScorer`, `ClipRegionSemanticScorer` e `AlphaClipSemanticScorer` compartilham a
regra de cosseno e pareamento em `_scoring_base.py`. As torres de texto `clip_text.py` e
`alphaclip_text.py` pertencem a scoring: codificam a hipótese para compará-la à feature visual.
Elas declaram o espaço da torre visual correspondente e usam o contrato público visual.

O template aceita `{hypothesis}` exatamente uma vez, sem outros campos. Template, scorer,
fingerprint de configuração e espaço de embedding entram no manifest.

Cosseno permanece em `[-1, 1]`; score ausente permanece ausente. Comparação entre espaços,
ponderação com outros sinais e calibração exigem regras científicas explícitas. O writer atual
persiste scores regionais e exige feature de região ancorada na mesma região do claim.

## Artifact

O schema atual é `0.2.0`. `AtomicRunDirectory` publica todos os arquivos contratuais por
staging e rename, com inventário de hashes. Writer e reader devolvem manifest tipado com
inventário e contagens de observações, claims e scores. Reexecução cria outra identidade.
Nenhum downstream depende de diagnósticos humanos.

Os testes de arquitetura verificam imports públicos, ausência de runtime/ML no núcleo e
as dependências permitidas. Referências: [fluxo](data-flow.md), [código](code-map.md) e
[testes](testing.md).
