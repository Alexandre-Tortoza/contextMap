# Testes de semantic_fusion

Os testes vivem em `tests/semantic_fusion/`. Cada responsabilidade do [code-map](code-map.md) tem um teste correspondente, no nível mais estreito que protege o comportamento.

## Responsabilidade → teste

| Responsabilidade | Arquivo | Nível |
| --- | --- | --- |
| Contrato de `FusionSupport` | `test_fusion_support_contract.py` | unit |
| Contrato de `EvidenceContribution` | `test_evidence_contribution_contract.py` | unit |
| Contrato e correlação de `FusedEvidence` | `test_fused_evidence_contract.py` | unit |
| Contratos de `QualityWeighting` | `test_quality_weighting_contract.py` | unit |
| Agrupamento por observação física | `test_physical_observation_grouping.py` | unit |
| Política de suporte: limiares, merge, exclusão, identidade | `test_fusion_support_construction.py` | unit |
| Poda de pares do motor de sobreposição | `test_fusion_support_construction.py::test_only_observations_that_share_geometry_are_compared` | unit (patcha `geometry_overlap.jaccard_overlap`) |
| Suportes como partição das observações | `test_support_partition_properties.py` | property-based (Hypothesis) |
| Política baseline de acumulação | `test_baseline_evidence_accumulation.py` | unit |
| Regras de incerteza | `test_uncertainty_preservation.py` | unit |
| Fator de interação, agregado e com proveniência | `test_interaction_factor.py` | unit |
| Atributos observados, sem agregação | `test_observed_attributes.py` | unit |
| Canais declarados e suas identidades | `test_evidence_channels.py` | unit |
| Política ciente de qualidade | `test_quality_aware_accumulation.py` | unit |
| Codecs JSON, round-trip e recusa de registro adulterado | `test_fusion_serialization.py` | unit |
| Artifact: escrita atômica, leitura por identidade, métricas, integridade, bytes gravados | `test_fusion_run_artifact.py` | integration |
| Avaliação da fusão | `test_fusion_evaluation.py` | integration |
| Boundaries e direção de dependências | `tests/architecture/test_boundaries.py` | architecture |
| Escopo de release e versão de schema | `tests/packaging/test_release_scope.py` | contract |
| Composição e despacho de política pelo runtime | `tests/runtime/test_runtime_composition.py`, `test_runtime_catalog.py` | integration |
| Caminho completo da Solution 1 | `tests/end_to_end/test_runtime_chain.py`, `tests/evaluation/test_end_to_end_scenario.py` | end-to-end |

Os builders (`fusion_builders.py`, `evidence_builders.py`, `observation_builders.py`, `quality_builders.py`, `geometry_fake.py`, `rich_fusion.py`, `fusion_run_fixtures.py`, `fusion_eval_fixtures.py`) constroem evidência de domínio **real**, não mocks: o único stand-in é `InMemoryGeometrySource`, que substitui o mapa persistido. Mocks substituem dependência cara, nunca comportamento que deve ser testado com tipos de domínio.

## Invariantes que nunca podem quebrar

Cada item abaixo tem teste. Quebrar qualquer um é mudança científica, não refatoração.

| Invariante | Protegido por |
| --- | --- |
| Toda coleção é ordenada e única; evidência equivalente codifica idêntica | contratos + `test_the_persisted_run_matches_the_recorded_bytes` |
| Score `None` é não pontuado, nunca zero | `test_fusion_serialization.py`, `test_baseline_evidence_accumulation.py` |
| Qualidade de observação nunca é `SupportSignal` | `test_quality_aware_accumulation.py::test_claim_confidence_and_scores_never_enter_the_weight` |
| Similaridade de cosseno mantém a faixa nativa `[-1, 1]` | `test_evidence_contribution_contract.py` |
| N inferências sobre um frame contam como uma observação física | `test_physical_observation_grouping.py`, `test_quality_aware_accumulation.py` |
| Contradição exige dois frames físicos distintos | `test_uncertainty_preservation.py::test_disagreement_between_runs_over_one_frame_is_ambiguity_and_not_contradiction` |
| Empate compara contagens, nunca scores | `test_uncertainty_preservation.py::test_a_tie_never_compares_scores` |
| Abstenção não é hipótese nem evidência contrária | `test_uncertainty_preservation.py::test_abstention_is_not_a_hypothesis_and_not_negative_evidence` |
| Nenhum label é abstenção sem a política dizer | `test_uncertainty_preservation.py::test_without_configured_abstention_labels_unknown_is_an_ordinary_label` |
| Observações são uma partição: cada uma em um suporte ou excluída | `test_support_partition_properties.py` |
| O resultado não depende da ordem de nenhuma entrada | `test_uncertainty_preservation.py::test_uncertainty_does_not_depend_on_the_order_of_the_evidence`, `test_quality_aware_accumulation.py::test_the_weights_are_reproducible_and_do_not_depend_on_order` |
| A ponderação não altera hipótese, contribuição nem incerteza | `test_quality_aware_accumulation.py::test_uncertainty_and_abstention_follow_the_baseline_settings` |
| Observação fraca não é descartada | `test_quality_aware_accumulation.py` |
| Estrutura 3D é listada uma vez e nunca ponderada | `test_quality_aware_accumulation.py::test_static_structure_is_listed_once_and_never_weighted` |
| Um run guarda uma política só | `test_fusion_run_artifact.py` |
| Registro adulterado é recusado, não confiado | `test_fusion_serialization.py` |
| O artifact reabre sem NumPy, ROS ou runtime de modelo | `test_fusion_run_artifact.py::test_a_persisted_run_reopens_without_numpy_or_any_model_runtime` |
| O writer não guarda linha das tabelas de índice | `test_fusion_run_artifact.py::test_the_writer_keeps_no_row_of_the_index_tables` |
| Só pares que compartilham geometria são comparados | `test_fusion_support_construction.py::test_only_observations_that_share_geometry_are_compared` |

## Teste de bytes gravados

`test_the_persisted_run_matches_the_recorded_bytes` compara o digest de **todo arquivo contratual** do run com um valor registrado. É a rede de segurança de qualquer refatoração desta capability: ele falha se o layout, a ordem das chaves, os separadores, os offsets dos índices ou qualquer campo mudarem. Uma refatoração estrutural deve mantê-lo verde sem tocar no valor registrado.

`__DEBUG__/` fica fora desse digest por contrato, então mudanças em `run_artifact/debug.py` precisam de inspeção explícita.

## Red → Green → Refactor nesta capability

Comportamento determinístico — contrato, transformação, serialização, ordem, linhagem — segue o ciclo normal: teste que falha pelo motivo esperado, implementação mínima, refatoração.

Uma mudança **estrutural ou documental** não inventa um `Red` artificial: os contratos já estão protegidos pelos testes acima, e o critério é que o conjunto de testes que falham antes e depois seja idêntico, com o teste de bytes gravados verde.

Uma mudança na **qualidade científica** da fusão (um critério de suporte diferente, uma regra de equivalência de label, uma nova forma de peso) não tem `Red` unitário: ela exige uma avaliação executável declarada antes da implementação — hipótese, baseline, variável alterada, dataset, métrica e critério de aceitação ou rejeição. Ver [avaliação de Semantic Fusion](../../evaluation/docs/semantic_fusion.md).

## Limitação atual da verificação

A qualidade da fusão nunca foi medida contra anotação: `corridor-02` roda de ponta a ponta, mas sem ground truth. Os testes garantem **consistência, preservação de evidência e reprodutibilidade**, não acerto semântico. Nenhuma decisão entre a política baseline e a ciente de qualidade pode ser tomada com o que existe hoje.
