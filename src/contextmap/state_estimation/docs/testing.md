# Testes de State Estimation

Os testes da capability ficam em `tests/state_estimation/`; as regras de fronteira, em `tests/architecture/`.

## Responsabilidade → teste

### Unit

| Responsabilidade | Teste |
| --- | --- |
| contratos de pose e trajetória, invariantes | `test_state_estimation_models.py` |
| lookup, interpolação, rejeições, resumo temporal | `test_state_estimation_lookup.py` |
| frame graph estático e laços | `test_state_estimation_frame_graph.py` |
| preflight, requisitos, relógios, `calibration_identity` | `test_state_estimation_preflight.py` |
| estatísticas de movimento | `test_state_estimation_motion.py` |
| regras de publicação compartilhadas (diagnóstico de gap) | `test_state_estimation_pose_validation.py` |
| serialização de pose e trajetória (inclusive linhagem auxiliar obrigatória) | `test_state_estimation_serialization.py`, `test_state_estimation_serialization_properties.py` (propriedades, Hypothesis) |
| `ExternalPoseEstimator` | `test_state_estimation_external_pose.py` |
| `FastLioEstimator`, com um `FastLioRunner` injetado | `test_state_estimation_fast_lio.py` |
| lado host do FAST-LIO (bag, job, parser, processo substituto) | `test_state_estimation_fast_lio_process.py` |
| wrapper de implantação, com um runtime ROS substituto | `test_state_estimation_fast_lio_wrapper.py` |
| receita da imagem Docker do FAST-LIO | `test_state_estimation_fast_lio_image_recipe.py` |

### Integration

| Fronteira | Teste |
| --- | --- |
| backend → artifact: escrita, leitura, inventário, debug, integridade | `test_state_estimation_run_artifact.py` |
| preflight → backend → resultado (serviço) | `test_state_estimation_preflight.py` |
| runtime → capability (pose auxiliar, composição, reuso, foundation) | `tests/runtime/test_state_estimation_pose_bridge.py`, `tests/runtime/test_runtime_composition.py`, `tests/runtime/test_runtime_foundation.py` |
| consumers (lookup e frame graph em Geometric Mapping e Sensor Association) | `tests/geometric_mapping/`, `tests/sensor_association/` |

### Architecture

| Regra | Teste |
| --- | --- |
| dependências entre capabilities, imports por caminho interno, backend só no runtime | `tests/architecture/test_boundaries.py`, `test_runtime_boundaries.py` |
| port sem dependência do preflight; núcleo e `preflight/` sem importar backend; `models`/`requirements` sem dependência interna; wrapper sem `contextmap`; scanner que cobre `import`, `from … import` e imports relativos | `tests/architecture/test_state_estimation_structure.py` |
| `preflight/`: contratos e fatos sem importar outro módulo do pacote; avaliadores sem importar `service` nem uns aos outros; o resto da capability só acessa a fachada do pacote | `tests/architecture/test_state_estimation_structure.py` |

### End-to-end e avaliação

| O que | Onde |
| --- | --- |
| cadeia mínima com `ExternalPose` | `tests/end_to_end/test_runtime_chain.py` |
| comparação ExternalPose × FAST-LIO sob um mesmo relatório, com alinhamento explícito | `test_state_estimation_backend_evaluation.py` |
| qualidade da trajetória, ATE/RPE e perfil de referência | `tests/evaluation/test_state_estimation_evaluation.py`, `test_state_estimation_reference_profile.py` (pulados sem o dataset `corridor-02`, que não é versionado) |

A **qualidade científica** de um backend (ATE/RPE, comparação entre backends) é medida pela capability `evaluation`, não pelos testes unitários; ver [avaliação de State Estimation](../../evaluation/docs/state_estimation.md).

## Invariantes que nunca podem quebrar

- uma pose declara `parent_frame` e `child_frame` distintos, `translation_m` finita e `orientation` unitária `(x, y, z, w)`;
- uma trajetória tem timestamps estritamente crescentes, um único domínio de clock e frames iguais aos de cada pose;
- um timestamp em outro domínio de clock levanta `ClockDomainMismatchError`; domínios de clock nunca são comparados implicitamente;
- o lookup nunca extrapola, e uma pose interpolada é distinguível de uma estimada (`derived_from`);
- a covariância nunca é fabricada: `None` significa "sem incerteza reportada";
- uma run bloqueada pelo preflight nunca inicia o backend e nunca é persistida;
- um backend que falha nunca é substituído por outro;
- a orientação só é renormalizada dentro de uma tolerância declarada, e a conversão é registrada em `conversions_applied`;
- um artifact finalizado é imutável; `__DEBUG__/` nunca é lido como registro contratual e nunca entra no inventário;
- a linhagem auxiliar (`auxiliary_sequence_artifact_id`, `auxiliary_selection_id`) é obrigatória no registro persistido (`null` quando nenhuma sequência contribuiu).

## Como executar

```bash
pytest tests/state_estimation tests/architecture   # foco
make check                                          # gate canônico: ruff, mypy, pytest
```

`test_state_estimation_fast_lio_process.py` também roda nos jobs ROS da CI (`.github/workflows/ci.yml`).

## Política de TDD nesta capability

Mudança de comportamento determinístico segue Red → Green → Refactor. Uma reorganização puramente estrutural não ganha um Red artificial quando os contratos já estão protegidos; nesses casos a equivalência é demonstrada comparando a saída antes e depois (por exemplo, os bytes persistidos de um run artifact).
