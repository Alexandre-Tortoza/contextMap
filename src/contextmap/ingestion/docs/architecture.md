# Arquitetura de Ingestion

## Objetivo arquitetural

Ingestion isola detalhes de aquisição e produz contratos estáveis para o restante do ContextMap2.
A organização segue Clean Architecture de forma pragmática: a regra importante é a direção das
dependências, não a quantidade de camadas.

```text
runtime / composition
        |
        v
adapters concretos
        |
        v
contratos e regras de ingestion
```

Dados fluem da fonte para dentro da capability. Dependências de código apontam para contratos
canônicos, nunca de volta para ROS, bag SDKs ou runtime.

## Ownership

A capability possui semanticamente:

- `SourceObservation` e seus identificadores;
- `SourceProvenance`;
- políticas e diagnósticos de timestamp;
- sincronização de observações;
- calibração canônica e transforms estáticos;
- provenance de sequência;
- `SequenceArtifact`, seleção e replay.

Manter calibração em `ingestion/calibration/` é deliberado. Ingestion normaliza a calibração da
fonte e preserva a identidade usada pelas observações. Consumers como `sensor_association` podem
usar o contrato público sem que a calibração se torne uma capability global independente.

## Componentes

### Contratos e regras centrais

Arquivos na raiz de `ingestion/` definem observações, port de fonte, timestamp, validação,
sincronização, seleção, diagnostics e provenance.

Eles não devem depender de ROS, rosbags ou adapters concretos.

### Calibration

`ingestion/calibration/` possui o modelo canônico de câmera, transforms estáticos,
`CalibrationSet`, validação, serialização, identidade e refinamento.

A capability normaliza e preserva calibração. Aplicar a calibração para projetar pontos entre
sensores é responsabilidade de `sensor_association`.

### Sequence Artifact

`ingestion/sequence_artifact/` possui o formato persistente intermediário produzido pela ingestion.
Reader e writer fazem parte desse ownership porque o artifact é um contrato persistente da
capability, não uma integração incidental com filesystem.

### Adapters

`ingestion/adapters/` contém detalhes das fontes externas.

```text
ROS / pose file
      |
      v
adapter concreto
      |
      v
SourceAdapter + SourceObservation + CalibrationSet
```

Tipos nativos de ROS não podem atravessar essa borda.

## API pública

`src/contextmap/ingestion/__init__.py` é a superfície cross-capability.

Permitido em consumidores:

```python
from contextmap.ingestion import SourceObservation, CalibrationSet
```

Evitar em consumers externos quando existir export público:

```python
from contextmap.ingestion.canonical_observations import SourceObservation
```

O runtime é a exceção natural para adapters concretos, pois é o composition root responsável por
escolher a implementação.

## Dependências permitidas

```text
adapters -> contratos canônicos
adapters -> calibration
adapters -> SourceAdapter
sequence_artifact -> contratos canônicos
sequence_artifact -> calibration
sequence_artifact -> provenance / diagnostics
runtime -> API pública de ingestion
runtime -> adapters concretos durante composição
```

## Dependências proibidas

```text
contratos canônicos -> adapters
calibration -> adapters ROS
sequence_artifact -> ROS / rosbags
ingestion -> state_estimation
ingestion -> geometric_mapping
ingestion -> sensor_association
ingestion -> visual_perception
```

Ingestion pode produzir dados usados por essas capabilities, mas não deve importar suas regras para
tomar decisões em nome delas.

## Separação de responsabilidades

- decoding converte representação externa em representação canônica;
- timestamp normalization corrige explicitamente o tempo, quando configurado;
- validation detecta problemas e não repara dados silenciosamente;
- synchronization associa eventos e não altera seus timestamps;
- calibration representa relações estáticas e modelos de câmera;
- sequence artifact persiste evidência e metadata sem criar semântica;
- runtime compõe o fluxo e escolhe adapters.

Se uma função começar a realizar mais de uma dessas transformações, é sinal de provável violação de
ownership.

## Invariantes arquiteturais

- uma observação não conhece o formato da fonte que a originou além de provenance;
- nenhuma mensagem ROS aparece na API pública;
- ausência de modalidade é representada explicitamente, não preenchida com dados sintéticos;
- timestamp bruto não é perdido quando uma correção é aplicada;
- transforms dinâmicos não entram em `CalibrationSet.static_transforms`;
- artifact finalizado é imutável;
- leitura de artifact não exige acesso à fonte original;
- uma correção de calibração é versionada e rastreável, não um offset escondido em projeção.

