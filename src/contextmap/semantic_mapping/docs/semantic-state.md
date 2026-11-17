# Estado semântico da entidade

Este documento descreve `src/contextmap/semantic_mapping/semantic_state.py` (os contratos), `state_mapping.py` (a projeção de `FusedEvidence` e os atributos derivados), `primary_hypothesis.py` (a política da primária) e `interaction_radius.py` (a escala do fator de interação).

Uma entidade preserva a evidência semântica acumulada a montante em vez de colapsá-la em um par label/confiança. O estado é inspecionável **sem carregar nenhum backend de percepção ou de modelo**: só identidades, textos e números.

## Distinções preservadas

| Distinção | Como o contrato a mantém |
| --- | --- |
| hipótese × atributo | `EntityHypothesis` é o que a entidade pode ser; `EntityAttribute` é uma propriedade dela, com evidência e regra próprias. |
| alternativa × contradição | Alternativas competem entre si; uma contradição (`UncertaintyKind.CONTRADICTION`) é observações físicas distintas apoiando hipóteses incompatíveis. |
| desconhecido/abstenção × evidência negativa | `EvidenceStance.ABSTAINING` não é apoio nem evidência contra. |
| sem score × score baixo | `SupportSignal.value` `None` (não pontuado) é diferente de `0.0`. |
| propriedade observada × conhecimento externo | `AttributeOrigin.OBSERVED`/`DERIVED` exigem evidência; `EXTERNAL_KNOWLEDGE` é rotulado, exige uma derivação documentada e uma `ExternalKnowledgeSource` (fonte, versão e entrada), única forma de nomeá-la. |

## Contratos

### `EntitySemanticState`

| Campo | Significado |
| --- | --- |
| `hypotheses` | Todos os candidatos (`EntityHypothesis`), ordenados por evidência fundida e hipótese; vazio quando nenhuma vista produziu claim. |
| `ambiguity_state` | `AmbiguityState`, **sempre** o que `derive_ambiguity_state` calcula dos registros: o contrato recusa outro valor. |
| `provenance` | `SemanticStateProvenance`: a regra de mapeamento, a política do primário e o mínimo de suporte que ela exigiu (`min_supporting_physical_observations`, `None` quando a política não tem mínimo). |
| `primary_hypothesis` | `EntityHypothesisRef` ou `None`: conveniência, só quando o estado é não ambíguo e o mínimo de suporte da política, se houver, é atingido. |
| `attributes` | `EntityAttribute` ordenados por nome, valor, origem e derivação. |
| `uncertainty` | `EntityUncertainty`: cada registro de incerteza da fusão, com a evidência fundida que o reportou. |
| `withheld_primary` | `WithheldPrimaryHypothesis` ou `None`: por que um estado não ambíguo não expõe primária (a hipótese e os frames físicos distintos que a sustentam, abaixo do mínimo). `None` quando o `ambiguity_state` já explica a ausência ou quando há primária. |

Propriedades: `primary`, `alternative_hypotheses` (todas menos a primária) e `conflicts` (só as contradições).

Os campos `evidence_refs[]` e `fused_evidence_refs[]` do desenho conceitual não se repetem aqui: a evidência de cada claim vive em `EntityHypothesis.evidence` e as referências à evidência fundida em `Entity.evidence` (`EntityEvidenceLinks`), uma única fonte de verdade.

### `EntityHypothesis`

`fused_evidence_id` + `hypothesis_id` (identidade dentro da entidade), o `label` verbatim, a `evidence` (`HypothesisEvidence` de Semantic Fusion: claim, stance, papel e sinais tipados) e `supporting_physical_observations`: os frames físicos distintos cujas claims a sustentam, ordenados, exatamente como a fusão os conta (`FusedEvidence.supporting_physical_observations`; inferência repetida sobre um frame conta uma vez). Esse campo existe para que o suporte independente continue contável **depois** da resolução de identidade, sem a evidência fundida à mão (a crença pós-resolução de Entity Resolution o consome). Ele só fica vazio numa entidade materializada antes de o campo existir: vazio significa "suporte não registrado", nunca "nenhum frame", e o registro o omite nesse caso. A propriedade `label_key` devolve a chave tipográfica sob a qual a fusão agrupou as claims da hipótese (NFKC, casefold, espaços colapsados). Uma hipótese exige ao menos uma evidência que a suporte.

### `AmbiguityState`

| Valor | Quando |
| --- | --- |
| `conflicting` | Há uma contradição entre observações físicas. |
| `ambiguous` | Há ambiguidade ou quase empate, ou várias hipóteses de uma mesma evidência fundida sem nada que as decida. |
| `insufficient_evidence` | Nenhuma hipótese, ou um registro de evidência insuficiente. |
| `unambiguous` | Existe uma hipótese e nada compete com ela. |

A precedência é a da tabela, de cima para baixo.

### `EntityAttribute`

`name`, `value`, `origin`, `derivation_id` (regra ou estágio versionado que produziu o valor), `evidence` (`EvidenceReference` de contribuição e claim) e `support` (sinais tipados próprios, quando o produtor os emitiu; vazio significa "sem sinal", não suporte zero). Um atributo observado ou derivado sem evidência é recusado.

## Regras de mapeamento (`semantic_state_from_fused_evidence`)

Versionadas: `fused-evidence-semantic-state-v1`.

1. **Nada é descartado.** Toda hipótese, com toda a sua evidência (incluindo alternativas, claims ambíguas ou conflitantes, abstenções e sinais sem score), e todo registro de incerteza voltam exatamente como foram fundidos.
2. **Primária só quando justificada** (`unambiguous-single-hypothesis-v1`): exatamente uma hipótese e nenhuma competição. Caso contrário não há primária, e a entidade não recebe um label que a evidência não sustenta. O contrato recusa uma primária exposta enquanto o estado não é `unambiguous`, porque as alternativas ficariam escondidas.
3. **Mínimo de suporte, opcional** (`unambiguous-single-hypothesis-v2`, `MinimumSupportPrimaryPolicy`): a v1 mais um gate, a hipótese só é exposta quando ao menos `min_supporting_physical_observations` frames físicos distintos a sustentam (`FusedEvidence.supporting_physical_observations`). Abaixo do mínimo o estado continua `unambiguous`, sem primária e sem `class`, e `withheld_primary` diz por quê. O default 1 não filtra nada e reproduz a v1; a v1 continua sendo o default. A decisão, as opções e o que falta para escolher um mínimo acima de 1 estão em [`primary-hypothesis-policy.md`](primary-hypothesis-policy.md).
4. **Um único atributo derivado da primária.** `class`, derivado da hipótese primária (`primary-hypothesis-label-v1`, origem `derived`), citando exatamente as claims que a suportam.
5. **Atributos declarados pelas claims** (`claim-attribute-v1`, origem `observed`). Cada par `(nome, valor)` que as claims de suporte declararam, e que a fusão guardou em `FusedHypothesis.observed_attributes` (a `category` da claim sob o nome `category`, e `color`, `material`, `description`…), vira um `EntityAttribute` que cita exatamente essas claims, **com ou sem primária**. Um valor declarado em hipóteses diferentes é uma propriedade só, citando todas as claims; valores diferentes de um nome são atributos diferentes e **nenhum vence** (sem agregação para um valor único, sem ontologia fechada). O `interaction_factor` não entra aqui: tem tratamento tipado próprio na fusão e vira, na materialização, o atributo derivado `interaction_radius_m`, sob a política versionada `interaction-radius-from-factor-v1` (`InteractionRadiusPolicy`: `factor_min = 0.1`, `factor_max = 5.0` e regra de extensão `half_max_extent`, `raio = 0.5 * max(extent) * clamp(fator)`). O `derivation_id` do atributo é `<id da política>@<fingerprint>` e a evidência cita **somente** as claims que declararam o fator, não todas as que sustentam a primária. O raio continua uma hipótese semântica derivada, não uma medida. Nenhuma propriedade de senso comum, enriquecimento de ontologia ou conhecimento externo é adicionado.
6. **Suporte por hipótese.** Cada `EntityHypothesis` recebe `supporting_physical_observations` da evidência fundida.

O mapeamento é determinístico e não reexecuta percepção nem fusão.

## Validação

`tests/semantic_mapping/test_entity_semantic_state.py` usa **evidência fundida real**, construída pela acumulação baseline de Semantic Fusion (não registros feitos à mão), e cobre alternativas, contradição, quase empate, abstenção, apenas abstenções, vista sem claims, score ausente × zero, preservação integral, determinismo, persistência, as invariantes do contrato e a política da primária com mínimo de suporte (diferencial v1 × v2 com o default, 1 frame × 2 frames, inferência repetida, retenção explícita e round-trip).
