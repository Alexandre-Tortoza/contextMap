# Configuração de Ingestion

## Onde a configuração entra

Ingestion não deve procurar YAML, variáveis de ambiente ou parâmetros ROS por conta própria.
O runtime resolve configuração externa e constrói os contratos explícitos consumidos pela
capability.

```text
config externa / CLI
        |
        v
runtime
        |
        v
IngestionRequest
        |
        v
SourceAdapterConfig + TimestampPolicy + SynchronizationConfig
        |
        v
ingestion
```

Isso mantém as regras testáveis sem filesystem de configuração ou ROS.

## SourceAdapterConfig

| Campo | Obrigatório | Semântica |
| --- | --- | --- |
| `source_type` | sim | família do adapter, por exemplo `ros1_bag` |
| `path` | sim | fonte gravada |
| `topics` | sim como contrato; campos internos podem ser `None` | canais que serão lidos |
| `timestamp_clock_id` | não | identidade do clock de header compartilhado |
| `calibration` | não | calibração canônica fornecida externamente |
| `required_topics` | não | tópicos cuja ausência deve falhar |
| `window` | não | trecho do recording clock a ingerir |
| `extra` | não | configuração específica do adapter |

## SourceTopicMapping

Os campos atuais são:

- `rgb`;
- `camera_info`;
- `lidar`;
- `imu`;
- `pose`.

Um campo `None` significa que aquela modalidade não foi solicitada nessa fonte.

## Tópicos requeridos

`required_topics` transforma uma capacidade opcional em precondição daquela execução.

Exemplo conceitual:

```python
required_topics = frozenset({"rgb", "lidar"})
```

Se um deles não existir fisicamente na fonte, o adapter falha antes de emitir observações.

## SourceWindow

Uma janela é expressa no clock de gravação da fonte, não no `header.stamp` dos sensores.

```text
SourceWindow
    clock_id = <source>:recording_time
    start_seconds
    end_seconds
```

A separação é intencional porque recording time e event/header time podem possuir epochs ou offsets
diferentes. O adapter deve recusar uma janela de clock incompatível em vez de assumir equivalência.

## TimestampPolicy

A política temporal define:

- qual conceito de event clock está em uso;
- qual window clock está em uso;
- correção explícita opcional, atualmente `ConstantOffsetCorrection`.

O default não aplica correção. Uma correção nunca deve ser inferida apenas porque dois clocks
parecem próximos.

Quando aplicada, a forma persistida usa nanossegundos inteiros para preservar identidade e evitar
perda de precisão em epochs grandes.

## SynchronizationConfig

A configuração atual declara:

- `reference_modality`;
- `tolerance_nanoseconds`.

A sincronização ocorre depois da política de timestamp. Alterar tolerância não deve modificar o
timestamp das observações.

## Calibration

Calibração pode vir de duas origens:

1. metadata descoberta pelo adapter, como `CameraInfo`;
2. `CalibrationSet` fornecido pela configuração.

O merge precisa ser explícito e recusar conflito real para o mesmo sensor. Extrínsecos estáticos de
fontes cujo adapter não lê TF entram pelo `CalibrationSet` configurado.

## Adapter extra

`extra` existe para configuração específica que não pertence ao contrato comum.

Exemplo do pose file:

```yaml
extra:
  format: tum
  parent_frame: map
  body_frame: epson
  pose_role: ground_truth
```

Esses campos não devem migrar para `SourceAdapterConfig` enquanto não tiverem semântica comum a
múltiplos adapters.

## Regra para novas configurações

Antes de adicionar um campo compartilhado:

1. ele representa uma regra de ingestion ou apenas uma tecnologia?
2. mais de um adapter possui a mesma semântica?
3. o valor muda o contrato, uma policy ou só a construção da implementação?
4. ele precisa entrar em provenance/identidade para reprodução?

Configuração específica deve permanecer no adapter; política científica deve ter tipo e semântica
explícitos.

