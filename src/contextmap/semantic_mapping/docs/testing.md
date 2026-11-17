# Testes de semantic_mapping

Os testes vivem em `tests/semantic_mapping/`. Cada responsabilidade do [code-map](code-map.md) tem um teste correspondente, no nível mais estreito que protege o comportamento.

## Responsabilidade → teste

| Responsabilidade | Arquivo | Nível |
| --- | --- | --- |
| Contratos de `Entity`, `EntityReference` e `EntitySet`; escopo de identidade; codec | `test_entity_contract.py` | unit |
| Suporte 3D: resumos, estatísticas, orientação, diagnósticos, digest, verificação | `test_entity_geometry.py` | unit |
| Estado semântico: preservação integral, ambiguidade, primária v1 × v2, retenção | `test_entity_semantic_state.py` | unit |
| Atributos declarados pelas claims e suporte por hipótese | `test_entity_observed_attributes.py` | unit |
| Estado e histórico temporal, contagens distintas, clock único | `test_entity_temporal_state.py` | unit |
| Vínculos de evidência, integridade de referências, travessia da proveniência | `test_entity_evidence.py` | unit + integration |
| Raio de interação: clamp do fator e escala pela extensão medida | `test_interaction_radius.py` | unit |
| Materialização baseline, identidade, rejeições, fingerprint da configuração | `test_entity_materialization.py` | integration |
| Artifact: escrita atômica, leitura por identidade, manifest, índices, métricas, integridade, debug, bytes gravados | `test_semantic_mapping_run_artifact.py` | integration |
| Validação do run por `evaluate_semantic_mapping` | `test_semantic_mapping_evaluation.py` | integration |
| Boundaries cross-capability | `tests/architecture/test_boundaries.py` | architecture |
| Camadas internas e ownership do code-map | `tests/architecture/test_semantic_mapping_layering.py` | architecture |
| Composição, despacho e diretório do estágio | `tests/runtime/test_runtime_composition.py`, `test_runtime_catalog.py` | integration |
| Escopo de release e as duas versões de schema | `tests/packaging/test_release_scope.py` | contract |
| Consumidores a jusante sobre `Entity` | `tests/entity_resolution/`, `tests/spatial_relations/`, `tests/artifact/` | integration |
| Caminho completo da Solution 1 | `tests/end_to_end/test_runtime_chain.py`, `test_cross_stage_invariants.py` | end-to-end |
| Geometria não muda por causa dos estágios semânticos | `tests/end_to_end/test_semantics_do_not_change_geometry.py` | regression |

Os builders (`mapping_builders.py`, `mapping_fusion.py`, `mapping_geometry_fake.py`) constroem evidência de domínio **real**: `mapping_fusion.py` produz `FusedEvidence` executando a acumulação baseline de Semantic Fusion, não registros feitos à mão, e escreve runs de fusão reais em disco. Os únicos stand-ins são `InMemoryGeometrySource`, que substitui o mapa persistido, e os fakes de `FusedEvidenceSource`. Mocks substituem dependência cara, nunca comportamento que deve ser testado com tipos de domínio.

## Invariantes que nunca podem quebrar

Cada item abaixo tem teste. Quebrar qualquer um é mudança científica ou de contrato, não refatoração.

| Invariante | Protegido por |
| --- | --- |
| Toda coleção é ordenada e única; entidades equivalentes codificam idêntico | contratos + `test_the_persisted_run_matches_the_recorded_bytes` |
| Uma entidade nunca é só um label: alternativas, conflitos e abstenções sobrevivem | `test_entity_semantic_state.py`, `test_semantic_mapping_run_artifact.py` |
| Score `None` é não pontuado, nunca zero | `test_entity_semantic_state.py` |
| `ambiguity_state` é sempre o que os registros implicam | `test_entity_semantic_state.py` |
| Primária só quando justificada; sob a v2, só com suporte suficiente | `test_entity_semantic_state.py` (diferencial v1 × v2) |
| Uma primária retida registra o motivo em `withheld_primary` | `test_entity_semantic_state.py` |
| Inferência repetida sobre um frame conta uma observação física | `test_entity_temporal_state.py`, `test_entity_semantic_state.py` |
| Frames físicos e inferências são contados à parte em todo lugar | `test_entity_temporal_state.py`, métricas do artifact |
| O suporte geométrico é a autoridade; os resumos são deriváveis e checáveis | `test_entity_geometry.py` (`verify_geometry_summary`) |
| Suporte vazio não passa por geometria válida | `test_entity_geometry.py` |
| Não contar conectividade não é o mesmo que estar conectado (`None`, não `1`) | `test_entity_geometry.py` |
| Recalcular um resumo nunca muda o `EntityId` | `test_entity_materialization.py` |
| O `EntityId` não depende da ordem dos candidatos nem de irmãos rejeitados | `test_entity_materialization.py` |
| Dois suportes independentes permanecem duas entidades | `test_entity_materialization.py` |
| Não existe API de merge, split ou match | `test_entity_materialization.py` |
| As quatro causas viram rejeição; qualquer outro erro propaga | `test_entity_materialization.py` |
| Um suporte sem hipótese vira entidade `insufficient_evidence`, não rejeição | `test_entity_materialization.py` |
| Uma referência de outro semantic map é recusada mesmo com id existente | `test_entity_contract.py`, `test_semantic_mapping_run_artifact.py` |
| O digest da evidência fundida detecta conteúdo alterado; a sequência é comparada à parte | `test_entity_evidence.py` |
| A identidade de uma feature é a tripla run/resultado/feature | `test_entity_evidence.py` |
| Integridade **relata**, travessia **levanta** | `test_entity_evidence.py` |
| Um run guarda uma política só | `test_semantic_mapping_run_artifact.py` |
| Um run finalizado nunca é sobrescrito; uma escrita interrompida não deixa run visível | `test_semantic_mapping_run_artifact.py` |
| Registro adulterado é recusado, não confiado | `test_semantic_mapping_run_artifact.py` |
| `__DEBUG__/` nunca é fonte contratual e removê-lo não invalida o run | `test_semantic_mapping_run_artifact.py` |
| O artifact reabre sem NumPy nem runtime de percepção, fusão ou modelo | `test_semantic_mapping_run_artifact.py` (em subprocesso) |
| Uma entidade é lida sem carregar as outras | `test_semantic_mapping_run_artifact.py` |
| As duas versões de schema são independentes | `test_semantic_mapping_run_artifact.py`, `tests/packaging/test_release_scope.py` |
| Não há estado de merge, split ou resolução no artifact | `test_semantic_mapping_run_artifact.py` |
| Nenhuma camada interna importa a de cima | `tests/architecture/test_semantic_mapping_layering.py` |

## Teste de bytes gravados

`test_the_persisted_run_matches_the_recorded_bytes` compara o digest de **todo arquivo contratual** do run — `outputs/`, `metrics/` e o manifest sem `created_at` — com um valor registrado, em três casos: um run com entidades e rejeições, um run cuja geometria parcial rejeita candidatos, e um run vazio.

É a rede de segurança de qualquer refatoração desta capability: ele falha se o layout, a ordem das chaves, os separadores, os offsets dos índices, um fingerprint ou qualquer campo mudarem. Uma refatoração estrutural deve mantê-lo verde **sem tocar nos valores registrados**.

`__DEBUG__/` fica fora desse digest por contrato, então mudanças em `run_artifact/debug.py` precisam de inspeção explícita.

## Red → Green → Refactor nesta capability

Comportamento determinístico — contrato, projeção, serialização, ordem, linhagem, identidade — segue o ciclo normal: teste que falha pelo motivo esperado, implementação mínima, refatoração.

Uma mudança **estrutural ou documental** não inventa um `Red` artificial: os contratos já estão protegidos pelos testes acima, e o critério é que o conjunto de testes que falham antes e depois seja idêntico, com o teste de bytes gravados verde e os fingerprints fixados intactos.

Uma mudança na **qualidade científica** do mapa — outro critério de primária, outra regra de resumo geométrico, outra escala para o raio de interação — não tem `Red` unitário: ela exige uma avaliação executável declarada antes da implementação, com hipótese, baseline, variável alterada, dataset, métrica e critério de aceite. Ver [avaliação de Semantic Mapping](../../evaluation/docs/semantic_mapping.md).

## Como rodar

```bash
pytest tests/semantic_mapping tests/architecture
pytest tests/semantic_mapping tests/architecture tests/entity_resolution tests/spatial_relations tests/artifact
make check
```

## Limitação atual da verificação

A qualidade semântica do mapa nunca foi medida contra anotação: o `corridor-02` roda de ponta a ponta, mas sem ground truth. Os testes garantem **consistência, preservação de evidência e reprodutibilidade**, não acerto semântico. Nenhuma decisão entre a v1 e a v2 da hipótese primária, ou sobre um mínimo de suporte acima de 1, pode ser tomada com o que existe hoje.
