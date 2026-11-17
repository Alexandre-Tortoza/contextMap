# Mapa do código de Ingestion

Este documento responde à pergunta "qual arquivo devo abrir para entender ou alterar esta
responsabilidade?". Ele descreve ownership, não apenas o conteúdo atual. Se uma nova função não
encaixar na responsabilidade declarada do arquivo, primeiro verifique se ela pertence a outro
módulo.

## Raiz da capability

| Arquivo | Responsabilidade | Entrada -> saída | Não deve possuir |
| --- | --- | --- | --- |
| `__init__.py` | fachada pública cross-capability | símbolos internos -> API estável | implementação de regra ou adapter |
| `canonical_observations.py` | tipos canônicos de imagem, LiDAR, IMU e pose externa | valores normalizados -> `SourceObservation` | ROS, I/O, sincronização |
| `recorded_source_adapter.py` | port e configuração comum de fontes gravadas | configuração -> contrato `SourceAdapter` | implementação ROS concreta |
| `source_timestamp_normalization.py` | políticas, correções e diagnóstico de clock | observação + política -> timestamp normalizado | sincronização entre modalidades |
| `source_observation_validation.py` | invariantes estruturais de observações e frames | observações -> lista de problemas | reparo silencioso |
| `observation_synchronization.py` | associação temporal determinística | observações -> grupos + diagnostics | correção de timestamp, interpolação |
| `sequence_observation_selection.py` | seleção/replay pelo índice do artifact | reader + seleção -> offsets/observações lazy | leitura de ROS |
| `sequence_observation_diagnostics.py` | sumários estruturais e frame graph | metadata canônica -> diagnostics | interpretação semântica |
| `sequence_provenance.py` | linhagem e identidade de conteúdo | fonte/config/código -> provenance | avaliação de qualidade |

## Calibration

| Arquivo | Responsabilidade | Entrada -> saída | Não deve possuir |
| --- | --- | --- | --- |
| `calibration/__init__.py` | fachada do subdomínio de calibração | símbolos internos -> API do pacote | regras novas |
| `calibration/calibration_set.py` | modelos de câmera, transforms, entries e `CalibrationSet` | valores canônicos -> contratos | ROS, filesystem |
| `calibration/calibration_set_validation.py` | invariantes estruturais de calibração/refinement | `CalibrationSet` -> problemas | correção automática |
| `calibration/calibration_content_identity.py` | digests determinísticos | valores canônicos -> SHA-256 | provenance descritiva na identidade |
| `calibration/calibration_refinement.py` | aplicar refinamento versionado à extrínseca alvo | base + refinement -> novo `CalibrationSet` | estimar refinement |
| `calibration/calibration_serialization.py` | JSON canônico e documento de refinement | contratos <-> registros | descoberta de calibração |

## Sequence Artifact

| Arquivo | Responsabilidade | Entrada -> saída | Não deve possuir |
| --- | --- | --- | --- |
| `sequence_artifact/__init__.py` | fachada pública do artifact | símbolos internos -> API do pacote | regra de I/O |
| `sequence_artifact/sequence_artifact_manifest.py` | schema, manifest e nomes de layout | manifest JSON <-> contrato | leitura de payload |
| `sequence_artifact/source_observation_serialization.py` | observação <-> index/payload | `SourceObservation` <-> registro + bytes | ROS |
| `sequence_artifact/sequence_artifact_writer.py` | escrita streaming e publicação atômica | observações/metadata -> artifact final | semântica downstream |
| `sequence_artifact/sequence_artifact_reader.py` | leitura e replay seletivo | artifact -> observações/metadata | acesso à fonte original |
| `sequence_artifact/sequence_artifact_integrity.py` | inventário e cross-references | artifact -> problemas de integridade | reparo do artifact |

## Adapters

| Arquivo | Responsabilidade | Entrada -> saída | Não deve possuir |
| --- | --- | --- | --- |
| `adapters/__init__.py` | package boundary dos adapters concretos | - | exports públicos desnecessários |
| `adapters/pose_file_source_adapter.py` | trajetória em arquivo -> pose externa | arquivo configurado -> `ExternalPoseMeasurement` | `PoseEstimate` |

## Adapters ROS

| Arquivo | Responsabilidade | Entrada -> saída | Não deve possuir |
| --- | --- | --- | --- |
| `adapters/ros/__init__.py` | package boundary da borda ROS | - | regra de domínio |
| `ros1_bag_source_adapter.py` | ler ROS 1 bag conforme `SourceAdapter` | bag -> observações/calibração/warnings/hash | lógica científica downstream |
| `ros2_bag_source_adapter.py` | ler ROS 2 bag conforme `SourceAdapter` | bag -> observações/calibração/warnings/hash | lógica científica downstream |
| `ros_adapter_operations.py` | fachada interna dos helpers compartilhados | helpers especializados -> namespace interno | novas responsabilidades genéricas |
| `ros_message_decoding.py` | desserialização, truncamento e header timestamp | bytes ROS -> mensagem/timestamp | decisão de modalidade |
| `ros_image_decoding.py` | Image -> `ImageObservation` | mensagem -> observação | percepção visual |
| `ros_lidar_decoding.py` | PointCloud2 -> `LidarObservation` | mensagem -> observação | voxelização/mapping |
| `ros_imu_decoding.py` | Imu -> `ImuObservation` | mensagem -> observação | integração/filtro |
| `ros_external_pose_decoding.py` | Odometry -> `ExternalPoseMeasurement` | mensagem -> medição externa | state estimation |
| `ros_calibration_extraction.py` | CameraInfo e merge de calibração | metadata ROS -> `CalibrationSet` | projeção 2D/3D |
| `ros_payload_normalization.py` | padding e byte order | bytes ROS -> bytes canônicos | política temporal |
| `ros_recording_window.py` | resolver janela no recording clock | `SourceWindow` + reader -> limites | comparar com header clock |
| `ros_source_content_hash.py` | hash incremental do trecho lido | mensagens cruas -> digest | passada extra sobre a fonte |

## Como usar este mapa em review

Ao encontrar código difícil de localizar, pergunte:

1. qual dado entra?
2. qual contrato sai?
3. qual regra semântica está sendo aplicada?
4. essa regra aparece na coluna "Responsabilidade" do arquivo atual?
5. o módulo está importando uma tecnologia que deveria estar na borda?
6. existe outro arquivo que já possui esse conceito?

Tamanho de arquivo é um sinal, não o critério de divisão. A divisão deve seguir razões independentes
de mudança e ownership.

