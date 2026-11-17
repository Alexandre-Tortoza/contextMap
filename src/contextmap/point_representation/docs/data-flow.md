# Fluxo de dados de Point Representation

Este documento segue os dados reais através da implementação. Cada seta existe no código; nada aqui
é etapa planejada.

Unidade e frame, válidos em todo o fluxo: **coordenadas em metros, no frame do mapa geométrico**. A
medição sensor-local de um ponto nunca é usada — `support.py` lê o `coordinates_m` autoritativo.

## Caminho principal

```
SpatialObservation (run de associação)        GeometrySource (mapa geométrico)
            │                                            │
            └──────────────┬─────────────────────────────┘
                           ↓
            ObservationSupportAnchorPolicy.select_centers()
                           ↓  tuple[GeometryReference, ...]  ordenada por geometry_id
            SupportExtractor.extract(center)
                           ↓  PreparedSupport
            PointEncoder.encode(prepared)
                           ↓  EncodedVector
            RepresentationService._assemble()
                           ↓  EncodedRepresentation | FailedSupport
            PointRepresentationRunWriter.add()
                           ↓
            PointRepresentationRunWriter.finalize()
                           ↓  outputs/ + metrics/ + __DEBUG__/ publicados atomicamente
            PointRepresentationRunReader
                           ↓
            entity_resolution / evaluation
```

## 1. Seleção de centros

`center_selection.py` recebe as observações espaciais de um run de associação e a fonte de geometria.
Para cada observação, ancora **uma** representação: o membro do suporte de geometria daquela
observação mais próximo do centróide do suporte.

O centróide usa `math.fsum`, para que não dependa da ordem dos membros por erro de arredondamento. O
resultado é deduplicado por `geometry_id` e ordenado, então o mesmo run de associação sempre produz o
mesmo conjunto de centros.

É uma **heurística**, e a política diz isso: a âncora não pretende ser o centro do objeto, um ponto
saliente nem uma amostra representativa. Ela só faz as representações de um run seguirem o que a
câmera observou, a um custo limitado pelo número de observações em vez do tamanho do mapa. Limites
em [`center-selection.md`](center-selection.md).

Este é o **único** ponto do fluxo onde `sensor_association` entra. Depois daqui, nada a jusante
conhece observação, máscara, feature ou claim.

## 2. Extração do suporte

`support.py` resolve cada centro num vizinhança pela porta `GeometrySource` — nunca por um índice
concreto —, segundo a `SupportPolicy`:

| `SupportType` / `NeighborhoodMethod` | Como seleciona | `query_method` gravado |
| --- | --- | --- |
| `POINT` | só o próprio centro | `reference-lookup` |
| `NEIGHBORHOOD` / `RADIUS` | prefiltro por bounds, depois raio euclidiano | `bounds-query+euclidean-radius` |
| `NEIGHBORHOOD` / `K_NEAREST` | bounds com raio dobrando até ter k, depois k mais próximos | `bounds-query-doubling+euclidean-k-nearest` |

A ordem dos membros é contrato: centro primeiro, depois distância crescente ao centro, empate por
`geometry_id` crescente. O centro pertence ao próprio suporte mesmo que a fonte não o devolva.

Em seguida as coordenadas são preparadas de forma determinística e **registrada** — a
`CoordinatePreparation` (`CenteringMode`, `ScaleNormalization`) vai para o `PreparedSupport`, então um
consumidor sempre sabe o que o encoder realmente recebeu.

Saída: `PreparedSupport`, que carrega o `PointSupport` (quais referências, qual política, qual
`query_method`, as estatísticas do suporte) e as coordenadas locais.

## 3. Encoding

`service.py` entrega o `PreparedSupport` a **um** encoder selecionado e monta o contrato canônico.
Dois encoders implementam a mesma porta:

```
PreparedSupport
    ├─→ GeometricDescriptorEncoder          14 features de covariância, Python puro
    └─→ PTv3PointEncoder ──→ PTv3Runtime    vetor de 64 dims do backbone Pointcept
                                 ↓
                        voxelization → backbone → pooling
```

No caminho PTv3, o `PreparedSupport` é voxelizado (`voxelization.py`, numpy puro), passa pelo backbone
residente no device (`pointcept_runtime.py`), e as features por voxel são reduzidas a um vetor por
suporte (`center` ou `mean`). Nenhum tensor, objeto de SDK ou payload nativo atravessa a fronteira: o
que volta é `PTv3Inference`, e o que sai do encoder é `EncodedVector`.

## 4. Montagem e falha explícita

`RepresentationService` produz, por centro, exatamente um de dois resultados:

- **`EncodedRepresentation`** — a `PointRepresentation` canônica mais os valores do vetor;
- **`FailedSupport`** — com `FailureReason.UNENCODABLE_SUPPORT` (o encoder declarou que não
  consegue) ou `FailureReason.NON_FINITE_OUTPUT` (o resultado não é finito).

Nada é substituído por vetor zero ou default, e não há fallback para outro encoder. Uma violação de
contrato ou um crash de backend **para o run** em vez de produzir um resultado silenciosamente
degradado.

Componentes que o encoder declara indefinidos guardam um placeholder nulo, nunca o valor do encoder —
e `is_partial` expõe isso no contrato.

## 5. Persistência

`run_artifact/writer.py` acumula os resultados sob as invariantes do run (um resultado por centro, o
vetor cabe no espaço declarado, nada contradiz o espaço e o encoder do manifest), recusa publicar um
stream que não corresponde às métricas reportadas, e publica via `AtomicRunDirectory`.

```
manifest.json                                 não inventariado
README.md
outputs/representations.jsonl                 um registro por representação
outputs/representation-index.jsonl            representation_id → offset/length de byte
outputs/representation-spaces.json
outputs/geometry-representation-index.jsonl   map_id/geometry_id → representation_id
outputs/failed-supports.jsonl
outputs/payloads/vectors.{f32|f64}            binário, endereçado por linha
metrics/counts.json  support-size.json  norms.json  runtime.json
__DEBUG__/support-traces.jsonl                não contratual
__DEBUG__/representation-norms.csv            não contratual
__DEBUG__/component-statistics.json           não contratual, só em FULL
```

O registro de uma representação **nunca** embute seu vetor: ele referencia
`outputs/payloads/vectors.<sufixo>#<linha>`. Por isso um run abre e é lido por identidade com a
biblioteca padrão, sem numpy e sem o mapa geométrico.

`outputs/` e `metrics/` são inventariados com tamanho e SHA-256 no manifest; `__DEBUG__/` nunca é,
então removê-lo não invalida o run.

## 6. Leitura

`run_artifact/reader.py` abre um run finalizado e responde por identidade, sem carregar o resto:
o espaço, uma representação por id, uma representação pela geometria a que está ancorada, um vetor,
os suportes falhados, um registro contratual por caminho, e a verificação de integridade.

`__DEBUG__/` é recusado explicitamente na leitura de registro: um consumidor não pode, nem por
acidente, depender de diagnóstico humano.

## Caminhos paralelos

### Proveniência

Anexada em dois momentos, e nunca reconstruída depois:

| Onde | O que entra |
| --- | --- |
| `RepresentationProvenance`, por representação | `code_version`, `run_id`, o `PointSupport` com sua política e `query_method` |
| `PointRepresentationRunManifest`, por run | `sequence_name`, `run_id`, `run_index`, `map_id` e lineage do mapa, `EncoderIdentity` (adapter version, configuração resolvida, hash do checkpoint), `representation_space_id`, `center_selection_policy_id`, `association_context_id` (identidade opaca), `schema_version`, `created_at`, inventário com hash |

O manifest **referencia** o mapa geométrico; nunca o copia.

### Identidades

Três identidades determinísticas, computadas em lugares distintos:

```
representation_space_fingerprint(space)   comparability.py   sha256 do JSON canônico do espaço
representation_id_for(run_id, index)      models.py          "<run_id>--repr-<índice:09d>"
center_selection_id(map_id, centers)      run_artifact/manifest.py   sha256 do mapa + ids ordenados
```

A terceira é independente da ordem de propósito: é como uma avaliação prova que dois encoders foram
comparados sobre exatamente os mesmos centros.

### Configuração

```
TOML / JSON externo
    ↓
runtime: resolve, valida o DAG, escolhe os dois pontos de variação
    ↓
SupportPolicy, PTv3Config, PointceptBackbone — tipados, por construtor
    ↓
regra / backend
```

Nenhum módulo da capability abre arquivo de configuração, lê env ou consulta parâmetro de ROS. O
`PTv3Runtime` chega injetado por um provider externo ao repositório.

### Diagnóstico

`SupportStatistics` por suporte e `RepresentationMetrics` por run (requisitados, representados,
parciais, falhados por razão, tempo de extração, tempo de encoding) seguem o caminho contratual até
`metrics/`. Traces de suporte, normas por representação e estatística por componente seguem para
`__DEBUG__/`, amostrados, e param ali.

## Os dois caminhos de falha

| Falha | Vira | Efeito no run |
| --- | --- | --- |
| suporte inencodável, ou saída não finita | `FailedSupport` registrado em `outputs/failed-supports.jsonl` e contado nas métricas | o run **é publicado**, com a falha auditável |
| violação de contrato, espaço incompatível, crash de backend, stream que não bate com as métricas | exceção | **nada é publicado** |

## O que não existe neste fluxo

- nenhum label, claim, embedding visual ou escore semântico entra ou sai;
- nenhuma fusão entre observações, nenhuma entidade, nenhuma relação;
- nenhum `confidence` genérico: a capability produz vetor e identidade de espaço, não crença;
- nenhum consumidor a jusante do run artifact no pipeline canônico — a fusão não tem entrada
  `representation` e a configuração do runtime recusa esse canal. O run é saída derivada;
- nenhuma gravação na geometria: a porta `GeometrySource` é só de leitura.
