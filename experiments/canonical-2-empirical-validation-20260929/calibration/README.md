# Calibração reconstruída (fresh)

O `SequenceArtifact` canônico do corridor-02 (`720a486d…`), do qual a ingestão reprodutível
(`experiments/corridor-02-window-ingestion-20260927`) lê calibração, config e âncoras, **não
estava presente nesta máquina**. Como não há carregador público de YAML/Kalibr em
`contextmap.ingestion`, a calibração foi **reconstruída** a partir dos YAMLs do dataset, com uma
**nova identidade** (hashes de conteúdo diferentes do build canônico), mas **geometria idêntica**.

`manifest.json` registra a identidade completa (hashes dos YAMLs, MEI, transforms, sensores).

## Decisões (documentadas porque definem a identidade da calibração)

- **Modelo de câmera**: `MeiCameraModel` a partir de `corridor-02-Intrinsics.yaml`
  (`gamma1/2 -> fx/fy`, `u0/v0 -> cx/cy`, `xi`, `k1,k2,p1,p2`).
- **`sensor_id` = `sanitize_topic(<tópico do bag>)`**: `/camera_1/image_raw -> camera_1_image_raw`,
  `/velodyne_points -> velodyne_points`, `/imu/data -> imu_data`. É assim que o adapter ROS1 liga
  cada observação à sua entrada de calibração (`calibration_ids_by_sensor`).
- **Frames**: `epson` = IMU/corpo, `d` = câmera, `cmu_rc1_velodyne` = LiDAR (nomes do README da
  ingestão canônica).
- **Sentido dos extrínsecos**: as matrizes `*_to_imu` do YAML viram
  `RigidTransform(parent=epson, child=sensor)`, i.e. `p_epson = R·p_child + t`. Quaternion
  `(x,y,z,w)` obtido da rotação 3x3 via `scipy.spatial.transform.Rotation`.

## Consequência

A calibração tem identidade nova, então **não é byte-comparável** ao build canônico; por isso este
experimento é uma validação empírica com identidade própria, não um rerun do run canônico. A
geometria (intrínsecos MEI e transforms) é a mesma dos YAMLs do dataset.
