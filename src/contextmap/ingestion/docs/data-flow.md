# Fluxo de dados de Ingestion

## Visão geral

```text
fonte gravada
    -> adapter
    -> observação canônica
    -> timestamp normalizado
    -> validação
    -> sincronização
    -> SequenceArtifact
    -> seleção/replay
```

Cada etapa preserva a distinção entre evidência física de fonte e decisões de processamento.

## 1. Leitura da fonte

O runtime constrói um adapter concreto a partir de `SourceAdapterConfig`. O adapter inspeciona as
capacidades e lê somente os tópicos/configurações solicitados.

```text
ROS 1 bag / ROS 2 bag / pose file
                |
                v
          SourceAdapter
```

Nenhum objeto ROS deve sair dessa etapa.

## 2. Decoding por modalidade

### Imagem

```text
sensor_msgs/Image
    -> ros_image_decoding
    -> valida layout / padding / byte order
    -> ImageObservation
```

### LiDAR

```text
sensor_msgs/PointCloud2
    -> ros_lidar_decoding
    -> preserva PointField + layout binário
    -> LidarObservation
```

### IMU

```text
sensor_msgs/Imu
    -> ros_imu_decoding
    -> campos disponíveis + covariâncias
    -> ImuObservation
```

### Pose externa

```text
nav_msgs/Odometry ou pose file
    -> decoder / adapter
    -> ExternalPoseMeasurement
```

`ExternalPoseMeasurement` continua sendo uma medição de fonte. Ela não é convertida em
`PoseEstimate` dentro de ingestion.

## 3. Timestamp

O adapter preserva o `header.stamp` como timestamp da observação e o recording time do bag em
provenance quando disponível.

Esses clocks têm papéis diferentes:

```text
recording time
    -> seleção da SourceWindow

header stamp
    -> timestamp do evento canônico
    -> correção explícita opcional
    -> sincronização
```

Uma `TimestampPolicy` pode aplicar uma correção configurada. Quando isso ocorre, o timestamp bruto
é preservado em provenance antes da substituição.

## 4. Calibração

Calibração segue um fluxo paralelo às observações:

```text
CameraInfo + CalibrationSet configurado
              |
              v
       extraction / merge
              |
              v
       CalibrationSet canônico
              |
      +-------+-------+
      |               |
      v               v
calibration_id   SequenceArtifact
na observação
```

Intrínsecos descobertos na fonte podem ser combinados com extrínsecos estáticos fornecidos por
configuração. TF automático não faz parte do adapter atual.

## 5. Validação

Após decoding/normalização, a validação verifica invariantes como:

- layout de imagem e LiDAR;
- ordenação temporal;
- referências de frame/calibração;
- consistência requerida pelo contrato.

Validação detecta problemas. Ela não altera silenciosamente os dados para fazê-los passar.

## 6. Sincronização

`observation_synchronization.py` recebe observações já normalizadas.

```text
anchor modality
      |
      +-> nearest candidate: image
      +-> nearest candidate: lidar
      +-> nearest candidate: imu
      +-> nearest candidate: external_pose
```

Somente eventos no mesmo `clock_id` são comparáveis. A política atual não interpola dados.
Associações ausentes ficam explícitas e decisões são persistíveis como diagnostics.

## 7. Persistência

O writer recebe as observações canônicas e metadata da sequência.

```text
SourceObservation
CalibrationSet?
SequenceProvenance?
Diagnostics?
       |
       v
SequenceArtifactWriter
       |
       v
SequenceArtifact
```

Imagem e LiDAR usam payloads binários separados. IMU e pose externa permanecem inline no índice.

## 8. Replay

Depois da publicação:

```text
SequenceArtifactReader
        |
        +-> manifest
        +-> calibration
        +-> provenance
        +-> diagnostics
        +-> iter_index()
        +-> observation_at()
        +-> selection/replay
```

O reader não precisa do bag original. Seleções primeiro resolvem offsets pelo índice e somente
depois carregam payloads das observações realmente acessadas.

## 9. Warnings e falhas

Problemas recuperáveis de conteúdo que o adapter sabe classificar podem virar `SourceAdapterWarning`.
Violações estruturais ou configurações impossíveis devem falhar explicitamente.

A regra é não descartar informação em silêncio: skips, conversões e correções relevantes precisam
ser observáveis por warnings, provenance ou diagnostics.

