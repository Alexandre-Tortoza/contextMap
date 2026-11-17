# Janela reproduzível do corridor-02 para a campanha de ablação (issue #669)

Este experimento é um comando versionado que ingere uma janela do `corridor-02.bag` do mesmo jeito
que a ingestão canônica, e grava o que a campanha de ablação por janela (#670) precisa para
selecionar os frames: os dois `SequenceArtifact`s e o `window.json`.

> **Estado: preparado, ainda não executado sobre o bag real.** O comando e o seu contrato estão
> testados com um bag ROS 1 sintético e uma sequência canônica sintética
> (`tests/ingestion/test_corridor_window_ingestion.py`). O dataset não estava no ambiente em que
> isto foi preparado, então nenhum `window.json` real existe ainda.

## Problema

A campanha precisa do trecho de **2:30 a 2:35** do `corridor-02.bag`. A sequência ingerida que
existe (`720a486de8d44c16a9d3d2ff9fa7b1a4`) cobre outro trecho: a janela canônica de 90 s com 5 s
de folga de cada lado.

A ingestão canônica (`experiments/e2e-real-ingestion-corridor-02-20260923/`) chamou
`IngestionService` por um driver que nunca foi versionado. `contextmap ingest` não a reproduz: não
repassa janela nem política de timestamp e não carrega calibração externa
(`src/contextmap/runtime/docs/cli.md`, "Lacunas conhecidas").

## Por que um script de experimento, e não a CLI

`IngestionRequest.from_document` já aceita janela e política de timestamp, mas a CLI continuaria
sem os outros três pedaços:

- **Calibração.** A calibração canônica foi construída dos YAMLs do dataset pelo driver não
  versionado. Não existe carregador público de YAML/Kalibr em `contextmap.ingestion`, e o
  decodificador de calibração não é API pública. Escrever um parser novo exigiria uma decisão
  própria (mapeamento MEI, nomes de frames, sentido dos extrínsecos) e ainda poderia produzir outra
  identidade de calibração.
- **Janela a partir da primeira mensagem do bag.** Isso é conhecimento do bag, não da configuração
  do runtime.
- **A sequência de pose e o `window.json`.** São do experimento, não da CLI.

O script é a composition root deste experimento. Ele usa a API pública de `contextmap.ingestion` e
o serviço público de ingestão do runtime (`IngestionService`), e constrói os dois adapters
concretos (`ros1_bag` e `pose_file`) só ali. Nenhum código do pacote muda.

## O que o comando faz

### O que vem da ingestão canônica

Tudo o que torna a ingestão nova igual à canônica é **lido do `SequenceArtifact` canônico** pela
API pública (`SequenceArtifactReader.read_calibration()` e `read_provenance()`), nunca redigitado:

| O quê | De onde |
|---|---|
| Calibração (MEI da câmera e extrínsecos estáticos) | `calibration.json` do artifact canônico: exatamente a construída dos YAMLs, com os mesmos `content_hash` |
| Tópicos, tópicos obrigatórios, sincronização, validação (`on_problems="warn"`), `hash_source` | `ingestion_config` da provenance canônica |
| Relógio de header (`corridor-02-header`) | `timestamp_clock_id` da provenance canônica |
| Correção de offset constante | as duas âncoras gravadas na provenance canônica, com a correção **derivada de novo** por `ConstantOffsetCorrection.from_anchors` |

O offset gravado na provenance canônica nunca é lido: ela é anterior à #618 e guarda o offset em
segundos de ponto flutuante, enquanto as âncoras são exatas. Com as âncoras documentadas em
`experiments/e2e-real-ingestion-corridor-02-20260923/README.md`, o offset esperado é
`78621358412432135` ns. Se o artifact canônico não tiver calibração, relógio declarado ou correção
derivada de âncoras, o comando recusa antes de escrever qualquer coisa.

### A janela

- **Referência:** a primeira mensagem do bag, **de qualquer tópico**, no relógio de gravação do
  bag. Ela é lida do índice de chunks do bag (`rosbags`, `Reader.start_time`), sem decodificar
  mensagem. É outra referência que a da descrição da janela canônica ("a partir da primeira
  pose").
- **Instantes resolvidos:** `start_recording_ns = primeira mensagem + start_offset_s` e
  `end_recording_ns = primeira mensagem + end_offset_s` (fim exclusivo). O header bruto
  correspondente é `R - offset`, o relógio do `corridor-02-gt.txt` e do `selection.json` canônico.
  A correção leva o header bruto de volta a `R`. Por isso `start_ns`/`end_ns`, no relógio
  corrigido das observações, têm os mesmos números que os instantes de gravação. É o modelo de
  offset constante: cada sensor se afasta dele pela própria latência (câmera em média −18 ms, LiDAR
  −28 ms, IMU −139 ms, "Finding 1" da ingestão canônica).
- **Folga:** `padding_s` segundos lidos antes do início e depois do fim, no relógio de gravação
  (`SourceWindow`). O padrão é 5 s, o mesmo ±5 s da ingestão canônica: um frame na borda mantém
  contexto de sincronização e de pose dos dois lados. A folga precisa passar da latência do sensor
  mais afastado (~0,14 s, a IMU). Senão, uma observação da janela gravada fora dela não seria
  lida.
- **Recusas**, antes de ler ou escrever qualquer coisa (código de saída 2): valor não finito,
  início antes da primeira mensagem, janela vazia ou invertida, folga negativa, `window.json` já
  existente na raiz de saídas, bag sem mensagem.

### A sequência de pose

É o `corridor-02-gt.txt` inteiro, pelo adapter `pose_file`. Esse adapter recusa janela por
contrato (`src/contextmap/ingestion/docs/adapters.md`), e a ponte de pose de `state_estimation`
mescla a sequência auxiliar inteira de qualquer forma. A sequência usa o mesmo relógio, a mesma
correção e a mesma calibração da sequência do bag. Sem o mesmo relógio corrigido, a checagem de
plausibilidade da ponte recusaria a mesclagem. O que o arquivo não declara é declarado no script:
`format=tum`, `parent_frame=map`, `body_frame=epson`, `pose_role=ground_truth`.

### A qualidade da janela

O comando mede e reporta a qualidade, mas **nunca muda a janela**. É a definição da seleção
canônica (`experiments/semantic-fusion-corridor-02-20260921/selection.json`, `rule`): o maior
intervalo entre amostras consecutivas **dentro** de `[start_ns, end_ns)`, sem contar os intervalos
até as bordas. As amostras são as poses da sequência de pose e as varreduras de LiDAR da sequência
do bag, lidas só do índice. Os limiares são pose ≤ 605 ms e LiDAR ≤ 250 ms. `within_thresholds` é
falso quando um deles é excedido ou quando há menos de duas amostras na janela (o buraco fica
`null`). Fora dos limiares, o comando sai com 0 e avisa em stderr.

## `window.json`

Fica na raiz de saídas, ao lado de `sequences/`. É publicado inteiro ou não é publicado, e nunca
sobrescreve outro. Não tem caminho absoluto: só o nome do bag e locais relativos à raiz de saídas.

| Campo | Significado |
|---|---|
| `schema` | `contextmap.corridor-window/v1` |
| `bag` | nome do arquivo do bag |
| `bag_first_message_recording_ns` | gravação da primeira mensagem do bag |
| `start_offset_s`, `end_offset_s`, `padding_s` | o pedido |
| `clock_id` | o relógio de header declarado (`corridor-02-header`) |
| `start_ns`, `end_ns` | a janela sem folga, no relógio corrigido das observações; fim exclusivo |
| `start_recording_ns`, `end_recording_ns` | a mesma janela no relógio de gravação |
| `start_raw_header_ns`, `end_raw_header_ns` | a mesma janela no header bruto (o do arquivo de pose) |
| `sequence_artifact_id`, `sequence_location` | a sequência do bag; local relativo à raiz de saídas |
| `pose_sequence_artifact_id`, `pose_sequence_location` | a sequência de pose |
| `canonical_sequence_artifact_id` | o artifact canônico de onde vieram calibração, configuração e âncoras |
| `offset_correction` | `offset_ns` e as duas âncoras (`source_time`, `reference_time`) |
| `quality` | `max_pose_gap_ms`, `max_lidar_gap_ms`, os dois limiares e `within_thresholds` |

Os campos do contrato com a #670 são os demais. `start_recording_ns`, `end_recording_ns`,
`start_raw_header_ns`, `end_raw_header_ns` e `canonical_sequence_artifact_id` foram acrescentados
para registrar os instantes nos três relógios e a origem da calibração.

## Como executar

**Pré-requisitos:** `contextmap` com o extra `ros1` (`rosbags`), o dataset em `datasets/corridor-02/`
e o `SequenceArtifact` canônico em `outputs/` do checkout principal.

**Comando**, da raiz do checkout principal. O corridor-02 de 2:30 a 2:35, com a folga padrão:

```bash
python experiments/corridor-02-window-ingestion-20260927/scripts/ingest_window.py \
    --bag datasets/corridor-02/corridor-02.bag \
    --pose-file datasets/corridor-02/corridor-02-gt.txt \
    --canonical-sequence outputs/ingest-real/sequences/corridor-02/720a486de8d44c16a9d3d2ff9fa7b1a4 \
    --outputs-root outputs/corridor-02-window-20260927 \
    --start-offset-s 150 --end-offset-s 155 --padding-s 5
```

A raiz de saídas é uma escolha: o repositório não registra nenhuma. Ela só precisa não ter um
`window.json`. Cada execução publica artifacts novos, com identidades novas, em
`<raiz>/sequences/corridor-02/<id>` e `<raiz>/sequences/corridor-02-pose/<id>`, e grava
`<raiz>/window.json`. O progresso sai em stderr, e o `window.json` também é impresso em stdout.

Códigos de saída: `0` publicado (mesmo fora dos limiares de qualidade); `1` uma ingestão falhou
(o que já foi publicado continua, imutável, e nenhum `window.json` é gravado); `2` recusado antes
de qualquer escrita.

**Conferir no `window.json`:** `offset_correction.offset_ns == 78621358412432135`,
`clock_id == "corridor-02-header"`, `canonical_sequence_artifact_id ==
"720a486de8d44c16a9d3d2ff9fa7b1a4"` e a `quality`.

## O que volta para o Git

1. `<raiz>/window.json` → `experiments/corridor-02-window-ingestion-20260927/results/window.json`,
   sem edição. Ele nomeia os dois artifacts, os instantes e a qualidade, e não tem caminho
   absoluto.

Não voltam: o bag, o arquivo de pose e as duas sequências publicadas. Ficam em `outputs/`, fora do
Git, e são auditáveis pelos próprios `manifest.json` e `provenance.json`.

## Limites

- A correspondência entre relógios é o modelo de offset constante da ingestão canônica. A latência
  própria de cada sensor não é corrigida: é o que a folga cobre.
- A sequência de pose não tem janela: é o arquivo de referência inteiro.
- O comando depende do `SequenceArtifact` canônico presente na máquina. Sem ele, não há calibração
  canônica a reproduzir.
- Os limites da `SourceWindow` são segundos em ponto flutuante. Na época do corridor-02, isso tem
  resolução de ~0,24 µs, irrelevante diante da folga. Os instantes do `window.json` são inteiros
  exatos.
