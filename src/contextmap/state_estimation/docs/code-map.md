# Mapa do código de State Estimation

Este documento responde "qual arquivo devo abrir para entender ou alterar esta responsabilidade?". Ele descreve ownership, não apenas o conteúdo atual. Se uma função nova não encaixar na responsabilidade declarada do arquivo, discuta o ownership antes de colocá-la ali.

## Raiz da capability

| Arquivo | Responsabilidade | Entrada → saída | Não deve possuir |
| --- | --- | --- | --- |
| `__init__.py` | fachada pública cross-capability (63 símbolos, `__all__` explícito) | símbolos internos → API estável | implementação, backends concretos |
| `models.py` | contratos de pose e trajetória e suas invariantes (frames, unidade, ordenação, clock, gaps) | valores → `PoseEstimate`, `Trajectory`, `TimeBounds`, proveniência | lookup, serialização, regra de backend, ROS/NumPy |
| `requirements.py` | declaração do que um backend exige (modalidades, relações estáticas, modelo de câmera, frames dinâmicos) | declaração → `GeometryRequirements` | avaliação dos requisitos |
| `ports.py` | port `StateEstimator`, request/result, diagnósticos, erros de backend | — → contratos do boundary com backends | construção de backend, preflight |
| `service.py` | `execute_state_estimation`: preflight, backend, conferência do resultado, linhagem auxiliar | estimador + request → `StateEstimationOutcome` | escolha de backend, fallback, persistência |
| `preflight/` | avaliar frames, extrínsecos, relações e relógios; `calibration_identity` (pacote, ver abaixo) | observações + calibração + requisitos → `GeometryPreflightReport` | declaração de requisitos, interpretar ou reescrever calibração, reprojeção |
| `frame_graph.py` | grafo estático de frames: resolver `T_parent_child`, consistência de laços | `RigidTransform[]` → `ResolvedTransform`, `LoopInconsistency` | pose dinâmica, validar calibração numericamente (isso é do preflight) |
| `lookup.py` | `T_map_body(t)`: política, interpolação, rejeição explícita, resumo temporal | `Trajectory` + timestamp + `LookupPolicy` → `ResolvedPose` ou `RejectedLookup` | estimar offset de tempo, extrapolar, projetar pontos |
| `motion.py` | estatísticas de movimento por intervalo | `Trajectory` → `MotionSummary` | limiares de descontinuidade (são da avaliação) |
| `pose_validation.py` | regras de publicação compartilhadas por backends: valores finitos, renormalização registrada, gap, diagnóstico de gap | valores de pose → violação ou pose canônica | regras de frame, clock ou ordenação (dependem da fonte); API pública |
| `serialization.py` | JSON de pose, trajetória e dos relatórios (preflight, movimento, qualidade, diagnósticos) | contratos ↔ registros JSON | layout de arquivos, manifest, I/O |
| `run_artifact.py` | layout, manifest, escrita atômica e leitura do `StateEstimationRunArtifact` | `StateEstimationOutcome` → diretório; diretório → `Trajectory`/poses | codificação de relatórios, escolha do diretório, alocar `run_id` |

## `preflight/`

| Arquivo | Responsabilidade | Entrada → saída | Não deve possuir |
| --- | --- | --- | --- |
| `preflight/__init__.py` | fachada interna do pacote: contratos, `run_geometry_preflight`, `calibration_identity` | — | lógica de avaliação |
| `preflight/models.py` | contratos e resultados: tolerâncias, achados, checagens, relatório, convenções | — → `GeometryPreflightReport` e seus componentes | lógica operacional, importar outro módulo do pacote |
| `preflight/facts.py` | resumir as observações por modalidade (frames, clocks, referências de calibração) | `SourceObservation[]` → `ObservationFacts` | avaliar requisitos ou clocks |
| `preflight/calibration.py` | calibração estática: identidade, checagens numéricas, consistência de laços, achados e avisos | `CalibrationSet` → `StaticCalibrationAssessment`, `PreflightFinding[]` | saber o que uma capability exige, reprojeção |
| `preflight/readiness.py` | os requisitos declarados em `requirements.py` estão satisfeitos pelos inputs? (modalidades, frames sem ambiguidade, caminho estático, modelo de câmera) | `GeometryRequirements` + fatos + grafo → achados e checagens | declarar requisitos, validar a calibração numericamente |
| `preflight/clocks.py` | cada modalidade exigida tem um clock explícito e compartilhado | requisitos + fatos → `ClockCheck[]` e bloqueios | converter ou comparar domínios de clock |
| `preflight/service.py` | `run_geometry_preflight`: encadear os passos e montar o relatório | observações + calibração + requisitos → `GeometryPreflightReport` | detalhes numéricos ou de requisito de cada passo |

`calibration`, `clocks` e `readiness` são avaliadores independentes: nenhum importa outro nem `service`; só `service` os combina. `requirements.py` continua na raiz porque o **port** o consome; `preflight/readiness.py` apenas verifica o que ele declara.

## `backends/`

| Arquivo | Responsabilidade | Entrada → saída | Não deve possuir |
| --- | --- | --- | --- |
| `backends/__init__.py` | marcar o pacote como interno | — | reexportar backends |
| `backends/external_pose.py` | validar `ExternalPoseMeasurement` e publicá-las como trajetória | `StateEstimationRequest` → `StateEstimationResult` | nomes de arquivo/frame de dataset, calibração, reordenar amostras |
| `backends/fast_lio.py` | lado canônico do FAST-LIO: validar request, resolver extrínseco, montar `FastLioJob`, publicar a saída do runner; port `FastLioRunner` | request → `StateEstimationResult` | ROS, processo, formato de arquivo do FAST-LIO |
| `backends/fast_lio_process.py` | lado host do contrato de troca: bag, `job.json`, processo sem shell com timeout, parser TUM | `FastLioJob` → `FastLioRunOutput` | regras de publicação de pose, montar `Trajectory` |
| `backends/fast_lio_wrapper.py` | wrapper de implantação, executado dentro do container ROS 1 | `job.json` + bag → `trajectory.tum` + `status.json` | importar `contextmap`; ser usado pela API Python |

```text
fast_lio_wrapper.py
→ deployment-side wrapper
→ executado dentro do container ROS 1 (Python 3.8)
→ não faz parte da API Python de state_estimation
→ não deve importar contextmap
```

## Onde colocar uma nova responsabilidade

| Se a mudança é sobre… | Altere |
| --- | --- |
| um novo campo ou invariante de pose/trajetória | `models.py` (+ `serialization.py` e `artifact.md`) |
| o que um backend precisa declarar | `requirements.py` |
| uma nova checagem numérica ou de consistência da calibração | `preflight/calibration.py` (+ `preflight.md`) |
| uma nova checagem de requisito de relação, modalidade ou modelo de câmera | `preflight/readiness.py` (+ `preflight.md`) |
| uma nova checagem de clock domain | `preflight/clocks.py` (+ `preflight.md`) |
| um novo resultado ou campo do relatório de preflight | `preflight/models.py` (+ `serialization.py`) |
| como o tempo é resolvido ou interpolado | `lookup.py` (+ `lookup.md`) |
| uma regra de publicação comum a vários backends | `pose_validation.py` |
| um novo backend | `backends/<nome>.py` e a factory em `runtime/composition.py`; nada no núcleo |
| um novo arquivo ou métrica no artifact | `run_artifact.py` (layout) e `serialization.py` (codificação) |
| onde o artifact é gravado ou como a config chega | `runtime`, não esta capability |

Uma regra que **não** pertence aqui: qualquer coisa que dependa de um backend concreto no núcleo, de ROS fora de `backends/fast_lio_process.py` e `fast_lio_wrapper.py`, ou de um threshold científico de outra capability.
