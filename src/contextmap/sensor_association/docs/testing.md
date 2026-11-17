# Testes de Sensor Association

Este documento mapeia **responsabilidade -> teste**. Ele serve para duas perguntas: "o que
protege esta regra?" e "onde ponho o teste da regra que estou escrevendo?".

Os testes vivem em `tests/sensor_association/`, com cinco módulos de builders compartilhados
(`projection_builders.py`, `perception_builders.py`, `dense_builders.py`, `run_builders.py`,
`association_builders.py`) que montam cenas determinísticas e pequenas. Eles existem para que
um teste de regra não precise construir meia capability antes de afirmar qualquer coisa.

## Por responsabilidade

| Responsabilidade | Arquivo | Nível |
| --- | --- | --- |
| contratos de evidência e suas invariantes | `test_sensor_association_models.py` | unit |
| ida e volta JSON, com revalidação | `test_sensor_association_serialization.py` | contract |
| três modelos de câmera: projeção, raio inverso, domínio de visão | `test_camera_models.py` | unit + contract |
| cadeia de preparo da imagem | `test_raw_to_prepared_transform.py` | unit |
| seleção de candidatos: esfera, fronteira exata, índices globais | `test_candidate_geometry.py` | unit |
| propriedades do recorte de alcance sob `RAY_RANGE` | `test_range_culling_properties.py` | property (Hypothesis) |
| o que o recorte pode e não pode mudar, contra o mapa inteiro | `test_candidate_equivalence.py` | regression científica |
| cadeia de projeção completa, linhagem e frame rejeitado | `test_frame_projection.py` | unit + integration |
| oclusão por suporte local de profundidade | `test_visibility.py` | unit |
| regra de máscara, sobreposição sem vencedor, índices e estatísticas | `test_region_membership.py` | unit |
| regra de caixa (`box-depth-component-v1`) e suas recusas | `test_box_region_membership.py` | unit |
| máscara de grounding refinada alimentando o mesmo caminho (#568) | `test_refined_region_membership.py` | regression |
| componentes de qualidade, ausência explícita e derivação | `test_observation_quality.py` | unit + contract |
| amostragem densa: nearest, bilinear, preflight, proveniência | `test_dense_feature_sampling.py` | unit |
| um backbone diferente pelo mesmo caminho neutro | `test_dense_sampling_across_backends.py` | contract |
| diagnósticos, achados, tolerâncias e varredura temporal | `test_association_diagnostics.py` | unit + integration |
| o serviço de ponta a ponta e o fingerprint da configuração | `test_association_service.py` | integration |
| sequência do request/upstreams e percepção validada antes de recusa de pose | `test_association_service.py` | regression |
| execução streaming: o frame é persistido e solto, não retido | `test_association_streaming.py` | integration |
| escalabilidade e os portões de memória do recorte | `test_association_scaling.py` | benchmark/gate |
| layout, linhagem, leitura por identidade e integridade do run | `test_association_run_artifact.py` | integration |
| erro de lineage após um frame escrito descarta a transação inteira | `test_association_run_artifact.py` | regression + integration |
| **identidades persistidas, fixadas como literais** | `test_association_identity_characterization.py` | regression |
| o harness de validação que lê o artifact pela API pública | `test_association_evaluation.py` | integration |

## Fora da capability

| O que protege | Onde |
| --- | --- |
| direção de dependências, fachada pública, isolamento de SDK | `tests/architecture/test_boundaries.py` |
| composição, pontos de variação e o estágio no DAG | `tests/runtime/test_runtime_composition.py`, `test_runtime_dense_channels.py`, `test_runtime_box_membership.py` |
| os adapters nomeados na matriz de capacidades continuam importáveis | `tests/evaluation/test_capability_matrix.py` |
| a cadeia real entre estágios | `tests/end_to_end/test_synthetic_chain.py`, `test_cross_stage_invariants.py` |
| a versão de schema exposta no escopo de release | `tests/packaging/test_release_scope.py` |
| os módulos de persistência não são ignorados pelo Git | `tests/packaging/test_repository_hygiene.py` |
| importação de todos os módulos da distribuição instalada | `tests/packaging/smoke_installed_package.py` |

## Invariantes que nunca podem quebrar

1. **Identidade global.** `geometry_support` nomeia elementos do mapa, nunca posições de
   array; o artifact persiste índices globais. Um índice que não cabe em 32 bits é recusado
   em vez de truncado.
2. **Evidência por referência.** `SpatialObservation` não carrega XYZ, embedding ou texto
   de claim. O sink recebe arrays intermediários no envelope de execução `FrameAssociation`.
3. **Ausência é explicada.** Todo candidato tem um `VisibilityState`; todo componente de
   qualidade ausente tem razão; toda região não avaliada tem `SkipReason`.
4. **Partição.** As máscaras de visibilidade atribuem cada ponto a exatamente um estado, e as
   contagens de `MembershipStatistics` fecham com `point_count`.
5. **Sobreposição sem vencedor.** Duas regiões podem suportar a mesma geometria.
6. **Métrica de profundidade declarada.** `DepthMetric` vem do modelo de câmera e é gravada;
   um modelo desconhecido levanta erro em vez de assumir uma.
7. **Máscara no espaço certo.** Uma máscara fora da imagem preparada é recusada, não
   reamostrada.
8. **Importável sem NumPy.** Importar os contratos e abrir um run não pode carregar NumPy, ROS
   ou SDK de modelo.
9. **Artifact imutável e atômico.** Uma escrita interrompida nunca parece um run; um run
   finalizado não é modificado; um schema diferente é recusado, nunca migrado.
10. **Identidade persistida é estável.** `configuration_fingerprint`, `schema_version`, as
    versões de definições e os bytes de cada arquivo contratual são fixados como literais em
    `test_association_identity_characterization.py`. Esse teste é a rede que permite mover
    código sem mover dado: ele falha se e somente se algo que um consumidor já persistiu
    mudou de significado.
11. **Lineage consistente.** Request, mapa, trajetória e resultado de percepção referenciam
    a mesma sequência. Percepção pertence à observação RGB do frame, mesmo se não houver
    pose aceitável. Incompatibilidades levantam erro antes de entregar o frame ao sink.

## Execução local

```bash
pytest tests/sensor_association tests/geometric_mapping tests/packaging/test_repository_hygiene.py \
  --cov=contextmap.sensor_association --cov-report=term-missing
make check
```

A suíte geométrica protege o ajuste de importação tardia necessário para que a fachada e o
reader de associação abram sem carregar NumPy. Os cálculos continuam usando NumPy quando
executados. Cobertura aponta caminhos não exercitados, mas não substitui os invariantes
científicos ou garante qualidade em todo dataset.

## Como escrever um teste aqui

Para mudança determinística de comportamento, **Red -> Green -> Refactor**: o teste precisa
falhar pelo motivo esperado antes da implementação.

Para refatoração puramente estrutural não se inventa um Red. O que se faz é o oposto: fixar o
que não pode mudar **antes** de mexer, como o teste de caracterização acima faz, e então mover
o código com a rede já no lugar.

Para mudança de qualidade científica (uma política de oclusão nova, outro critério de
candidato) não há Red unitário: antes de implementar, defina hipótese, baseline, variável
alterada, dataset, métrica e critério de aceitação ou rejeição, e deixe o experimento
executável. `test_candidate_equivalence.py` é o exemplo no repositório: ele não afirma que o
recorte é inofensivo, ele fixa exatamente o que o recorte pode e não pode mudar, incluindo o
par adversarial que de fato diverge.

Mocks substituem dependência externa ou cara, nunca um tipo de domínio. Não há SDK para
mockar nesta capability; o que se fakeia é uma fonte de geometria (`ArrayGeometrySource`) e um
sink (`CollectingSink`).
