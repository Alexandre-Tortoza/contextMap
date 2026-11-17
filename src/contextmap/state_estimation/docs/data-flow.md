# Fluxo de dados de State Estimation

Segue os dados reais através do código, na ordem em que acontecem. Nada aqui descreve etapas planejadas.

## Caminho principal

```text
SourceObservation[] + CalibrationSet?            (Ingestion, entregues pelo executor do runtime)
        ↓
StateEstimationRequest                           ports.py
        ↓
estimator.geometry_requirements()                requirements.py  → GeometryRequirements
        ↓
run_geometry_preflight(...)                      preflight/       → GeometryPreflightReport
        ├─ BLOCKED → GeometryPreflightError      service.py       (o backend nunca inicia)
        ↓ READY
estimator.estimate(request)                      backends/*       → StateEstimationResult
        ↓
_require_result_answers_request                  service.py       (id, sequência, seleção,
        ↓                                                          proveniência do estimador, frames)
_with_auxiliary_lineage                          service.py       (copia a linhagem auxiliar)
        ↓
StateEstimationOutcome (preflight + result)
        ↓
StateEstimationRunWriter.finalize                run_artifact.py + serialization.py
        ↓
StateEstimationRunArtifact (diretório imutável)
        ↓
StateEstimationRunReader.trajectory()            run_artifact.py
        ↓
TrajectoryLookup(trajectory).pose_at / pose_for_observation   lookup.py
        ↓
ResolvedPose | RejectedLookup                    → Geometric Mapping, Sensor Association
```

O preflight roda **dentro** de `execute_state_estimation`; nenhum backend é chamado se a run estiver bloqueada, e uma run bloqueada nunca é persistida (o writer recusa).

## Dentro de um backend

```text
ExternalPose                                       FAST-LIO
ExternalPoseMeasurement[]                          LidarObservation[] + ImuObservation[] + CalibrationSet
   ↓ _violation: frame, valores,                      ↓ _build_job: frames, clock, ordenação, cobertura da IMU,
   ↓ clock, ordenação                                 ↓ extrínseco T_imu_lidar via StaticFrameGraph.resolve
   ↓ SKIP → diagnóstico / FAIL → erro                 FastLioJob
   ↓ _publish: canonical_orientation                     ↓ FastLioRunner.run(job)
   ↓ gap_between → timestamp_gap_diagnostic              ↓   SubprocessFastLioRunner: bag + job.json → processo
Trajectory + diagnostics                                 ↓   (fast_lio_wrapper no container) → trajectory.tum
                                                      FastLioRunOutput
                                                         ↓ check_pose_values, canonical_orientation, gap_between
                                                      Trajectory + diagnostics
```

`pose_validation.py` é o ponto comum: os dois backends aplicam as mesmas regras de valor, renormalização e gap, e cada um mantém suas regras de frame, clock e ordenação.

## Caminhos paralelos

| Caminho | De onde vem | Para onde vai |
| --- | --- | --- |
| **Calibração** | `CalibrationSet` de Ingestion | `StaticFrameGraph` (preflight e FAST-LIO); `calibration_identity` vai para a proveniência da trajetória |
| **Proveniência** | request (`sequence_artifact_id`, `selection_id`, auxiliar), backend (`EstimatorProvenance`), `current_code_version()` | `TrajectoryProvenance` → `trajectory.json` e manifest |
| **Diagnósticos** | backends (`EstimationDiagnostic`) | `metrics/diagnostics.jsonl`, `diagnostic_counts` no manifest, `__DEBUG__/` no nível `full` |
| **Métricas** | `Trajectory` (`quality_summary`, `summarize_motion`) e relatório de preflight | `outputs/quality.json`, `metrics/motion.json`, `metrics/preflight.json`, `outputs/frame-summary.json` |
| **Configuração** | runtime → `ExternalPoseConfig`/`FastLioConfig` | backend; só o `configuration_fingerprint` chega à proveniência |
| **Debug** | trajetória e relatório | `__DEBUG__/` (nunca contratual, nunca inventariado) |

## Leitura de uma pose pelo consumidor

`StateEstimationRunReader` abre a run só pelo diretório. `pose(id)` e `pose_at(timestamp)` usam `outputs/pose-index.jsonl` para ler apenas os bytes da pose. Para pose mais próxima ou interpolada, o consumidor monta `TrajectoryLookup(reader.trajectory())` e escolhe uma `LookupPolicy`; o runtime constrói essa política a partir da configuração de cada capability consumidora.
