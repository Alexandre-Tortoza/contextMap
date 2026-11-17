# Política da hipótese primária e mínimo de suporte

Este documento registra a decisão sobre o mínimo de suporte da hipótese primária e descreve `MinimumSupportPrimaryPolicy` e `WithheldPrimaryHypothesis` (`primary_hypothesis.py` e `semantic_state.py`). As regras gerais do estado semântico estão em [`semantic-state.md`](semantic-state.md).

## Problema

A política `unambiguous-single-hypothesis-v1` expõe a hipótese primária, e dela deriva o atributo `class` (`primary-hypothesis-label-v1`), sempre que o estado é `unambiguous`. `derive_ambiguity_state` olha contradição, competição e ausência de evidência, mas **nunca quanto suporte** a hipótese tem. Assim:

```text
1 frame → 1 claim → nenhuma competição → unambiguous → primária + class
```

Isso está em tensão com o AGENTS §4: uma única observação não deve virar verdade persistente automaticamente. Uma detecção isolada, possivelmente espúria, recebe o mesmo status de primária que uma hipótese sustentada por dezenas de frames distintos.

Não é um bug: é uma política científica versionada que simplesmente não considera suporte. Os mitigadores continuam valendo (as hipóteses, os sinais e os registros de incerteza são preservados, e o `temporal_state` traz as contagens de frames físicos), mas a primária não exige nada disso.

## Opções avaliadas

| | Opção A: gate parametrizado | Opção B: flag de suporte único |
| --- | --- | --- |
| O que faz | Nova política `unambiguous-single-hypothesis-v2` com `min_supporting_physical_observations`: abaixo do mínimo não há primária nem `class`. | Mantém a exposição e marca no estado (ou na provenance da primária) quando o suporte é de uma observação física. |
| Tensão com o AGENTS §4 | Resolvida no contrato público: a regra que expõe a primária exige suporte independente. | Continua no contrato público: a observação única ainda vira primária, e o corte fica com cada consumidor. |
| Impacto | Muda quais entidades têm primária/`class` quando o mínimo é maior que 1 (valores); o parâmetro entra na provenance e na identidade da configuração. | Menos intrusiva: só acrescenta um campo. |
| Custo | Exige escolher um valor para o mínimo; um valor acima de 1 é tuning científico. | Nenhum valor a escolher, mas cada consumidor reinventa o seu corte. |

## Decisão: opção A

- `unambiguous-single-hypothesis-v2` é a v1 **mais** um gate: a hipótese de um estado `unambiguous` só é exposta como primária quando pelo menos `min_supporting_physical_observations` observações físicas distintas a sustentam.
- O parâmetro é configuração de execução (`MinimumSupportPrimaryPolicy`), validado na construção: um inteiro maior ou igual a 1 (um `bool`, um `float` ou um texto é recusado com `ValueError`). A regra que o interpreta é a política versionada, e o valor efetivo fica na provenance do estado e no fingerprint da configuração.
- **O default é 1, e 1 não filtra nada**: toda hipótese tem ao menos uma claim de apoio, portanto ao menos uma observação física. A v2 com o default reproduz a v1 exatamente, exceto pela identidade da política; um teste diferencial compara as duas sobre os fixtures de Semantic Mapping.
- **A v1 continua disponível e inalterada**: é o default de `EntityMaterializationPolicy` e do perfil canônico. Seu fingerprint, porém, mudou uma vez quando a regra do raio de interação (`interaction-radius-from-factor-v1`) passou a fazer parte dele; fora isso, a escolha v1 × v2 não o altera.
- **Escolher um mínimo acima de 1 é tuning científico** e está fora do escopo desta mudança. Ele exige o reference set anotado, com um split de tuning separado do de avaliação, e uma avaliação definida antes: hipótese (um mínimo `N` reduz primárias espúrias sem perder muitas corretas), baseline (v1), variável alterada (só `N`), seleção de dados (o split de tuning), métrica (por exemplo, precisão e cobertura do label primário contra a anotação) e critério de aceite. Nenhum valor acima de 1 é recomendado aqui.

## Como o suporte é contado

`FusedEvidence.supporting_physical_observations(hypothesis_id)`, consumido como está: os frames físicos distintos cujas claims **apoiam** a hipótese.

- várias inferências sobre o mesmo frame contam uma vez;
- uma claim sobre muitos pontos de geometria conta uma vez;
- abstenções não contam como apoio;
- nunca é contagem de inferências nem de claims, e não há ponderação por qualidade ou confiança.

## Como o estado registra a retenção

Quando a v2 retém a primária:

| Campo | Valor |
| --- | --- |
| `ambiguity_state` | Continua `unambiguous`: nada compete com a hipótese, e `derive_ambiguity_state` não muda. |
| `primary_hypothesis` | `None`. |
| `attributes` | Vazio: sem primária não há `class`. |
| `hypotheses`, `uncertainty` | Exatamente como fundidos. |
| `withheld_primary` | `WithheldPrimaryHypothesis`: a hipótese retida e os frames físicos que a sustentam, ordenados; o tamanho dessa lista é o suporte comparado com o mínimo. |
| `provenance` | `primary_policy_id = "unambiguous-single-hypothesis-v2"` e `min_supporting_physical_observations = N`. |

O contrato recusa um `withheld_primary` num estado que não é `unambiguous` (a ambiguidade já explica a ausência), junto com uma primária exposta, com uma hipótese desconhecida, sem mínimo na provenance ou com suporte que não está abaixo do mínimo; `Entity` recusa frames que a entidade não vincula. Com isso, sob a v2, a ausência de primária sempre tem um motivo explícito no estado: o `ambiguity_state` ou o `withheld_primary`.

### Por que não reutilizar `insufficient_evidence`

Reutilizar um registro `UncertaintyKind.INSUFFICIENT_EVIDENCE` foi considerado e descartado:

1. `EntityUncertainty` guarda os registros **como Semantic Fusion os produziu**, e a validação (`uncertainty_preserved`) exige que sejam exatamente os da evidência fundida. Um registro criado aqui misturaria a proveniência de duas capabilities.
2. Esse registro levaria o `ambiguity_state` a `insufficient_evidence` com uma hipótese presente. A jusante esse status significa "nenhuma hipótese é sustentada": o `ContextSemanticState` do `ContextMapArtifact` recusa `insufficient_evidence` com hipóteses, e Entity Resolution só o atribui a um estado resolvido sem hipóteses, então um membro assim sairia `unambiguous` da resolução. A mudança quebraria contratos a jusante, fora do escopo.
3. Ambiguidade (competição entre hipóteses) e suficiência de suporte são eixos diferentes; o gate não torna a hipótese ambígua.

## Seleção e identidade

```python
EntityMaterializationPolicy(
    geometry=GeometrySummaryPolicy(...),
    primary_support=MinimumSupportPrimaryPolicy(min_supporting_physical_observations=N),
)
```

`primary_support=None` (o default) é a v1. `EntityMaterializationPolicy.fingerprint()` inclui o id da v2 e o mínimo, então dois runs com mínimos diferentes nunca compartilham a identidade da configuração; a escolha v1 não acrescenta nada ao fingerprint além do que a configuração já carrega (inclusive a regra do raio de interação). `semantic_state_from_fused_evidence(evidence, primary_support=...)` aceita a mesma escolha, e `evaluate_semantic_mapping` refaz o estado sob a política oferecida.

O runtime compõe a v1 e o perfil canônico não muda. A v2 ainda não é selecionável pela configuração do runtime: um novo componente no catálogo mudaria toda configuração efetiva, inclusive a de `examples/v0.1.0` (digest da configuração efetiva e plano resolvido), que é congelada. Expor a v2 no runtime é uma decisão à parte, que só tem consumidor depois que um mínimo acima de 1 for escolhido pelo tuning.

## A primária por membro não chega ao mapa: a crença pós-resolução (C9)

Medido no frame 10:31 (`experiments/semantic-geometry-investigation-20260930`): 83 regiões → 41 entidades, **todas** com uma observação física, 27 delas com primária. Configurar a v2 aqui **não mudaria nada no artefato público**: a primária vive só em `EntitySemanticState`; `ResolvedSemanticState` (Entity Resolution) guarda a união das hipóteses dos membros sem primária, e o `ContextSemanticState` do mapa não tinha campo para ela. Além disso, uma primária por membro é decidida antes de a resolução juntar observações do mesmo objeto, então ela nunca vê o suporte multi-vista.

Por isso a decisão sobre "qual label a entidade é" passou para depois da resolução, numa regra versionada própria de Entity Resolution, `resolved-belief-min-support-v1` ([`resolved-entities.md`](../../entity_resolution/docs/resolved-entities.md#crença-pós-resolução-resolved-belief-min-support-v1)): sobre a união dos membros, conta as observações físicas **distintas** por chave de label (o mesmo `supporting_physical_observations` desta capability, agora gravado em cada `EntityHypothesis`) e registra `tentative`, `supported` (a única label que atinge o mínimo é a primária), `contested` ou `insufficient_evidence`. É essa crença, e não a primária por membro, que o `ContextMapArtifact` expõe (`support_status`, `primary_label`). A v1/v2 desta capability continuam decidindo só o estado da entidade antes da resolução e o atributo derivado `class`, que o mapa deliberadamente não copia.

## Persistência

O registro canônico da entidade passou de `entity serialization 0.1.0 → 0.2.0`: `semantic_state.withheld_primary` (`null` ou `{hypothesis, supporting_physical_observations}`) e `semantic_state.provenance.min_supporting_physical_observations` (`null` sob a v1). Os dois campos estão sempre presentes; o leitor recusa runs com `entity_schema_version` 0.1.0, sem migração (v0.x). Ainda na `0.2.0` não lançada, cada hipótese ganhou `supporting_physical_observations` (a lista de frames que a sustentam), **omitido** quando vazio: uma entidade materializada antes do campo continua legível, com o suporte "não registrado", e a regra de crença de Entity Resolution recusa explicitamente contá-la em vez de adivinhar.

## Validação

- `tests/semantic_mapping/test_entity_semantic_state.py`: teste diferencial v1 × v2 com o default sobre os cenários de mapeamento; 1 frame × 2 frames com mínimo 2; inferência repetida sobre um frame conta uma observação; abstenção não conta; estados não ambíguos não ganham `withheld_primary`; validação do parâmetro; invariantes do contrato; round-trip da provenance e da retenção.
- `tests/semantic_mapping/test_entity_materialization.py`: diferencial v1 × v2 sobre o run de fusão de três suportes, fingerprint da v1 fixado e o mínimo no fingerprint.
- `tests/semantic_mapping/test_semantic_mapping_evaluation.py`: um run que reteve a primária passa em todas as checagens sob a própria política.
