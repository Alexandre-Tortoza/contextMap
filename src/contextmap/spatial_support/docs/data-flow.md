# Fluxo de dados de Spatial Support

## Caminho principal

```text
ContextMapArtifact    VoxelAggregationArtifact    GeometricMapArtifact
        |                       |                          |
        '-----------------------+--------------------------'
                                v
                   1. derivation.py  (seleção e recusa)
                                v
                      EntityCellInput[]
                                v
                   2. service.py  (a junção)
                                v
                     SemanticSpatialSupport
                                v
                   3. run_artifact/writer.py
                                v
                  sete arquivos num diretório
                                v
                   4. run_artifact/reader.py
                                v
                 viewer, alinhando por aggregate_index
```

## 1. Seleção dos três upstreams

`derive_semantic_spatial_support_artifact` abre `VoxelAggregationArtifactReader`,
`ContextMapArtifactReader` e `GeometricMapArtifactReader` e compara três ids de mapa: o que o
ContextMap declara no seu `geometry_link()`, o do manifest do mapa bruto e o
`source_map_id` da agregação. Se os três não coincidirem, a derivação é recusada **antes de
ler geometria**, com os três ids na mensagem: juntar as entidades de um mapa à grade de outro
produziria suporte sem significado.

Mesmo com ids iguais, a derivação exige `verify_integrity(source=raw)` da agregação antes da
junção. Essa verificação confere os inventários e a identidade do mapa bruto consumido e
rederiva a lineage da agregação. Depois, a localização explícita passada ao leitor do
`ContextMap` confronta o digest da dependência geométrica registrada no seu manifest com o
digest do mesmo mapa bruto. Um mapa bruto diferente com o mesmo `map_id` é recusado em ambas
as relações.

O `ContextMap` é aberto com `verify_hashes=True`: todos os arquivos inventariados são
hasheados antes de qualquer entidade ser consumida. Assim, uma adulteração que preserve o
tamanho de `entities.jsonl` não pode ser interpretada sob o `content_identity` original.

Em seguida cada `Entity` do ContextMap vira um `EntityCellInput`, que exige ao menos uma
referência de geometria e todas de um único mapa. É a adaptação do contrato público de
`artifact` para a entrada da junção, e é a razão pela qual este módulo é de domínio e não só
de infraestrutura.

## 2. A junção

`build_semantic_spatial_support` chama `aggregation.membership(raw_geometry)` **uma vez**: um
vetor que, para cada índice de ponto bruto, diz qual agregado o resume (`-1` quando nenhum).
Para cada referência de cada entidade, `geometry_index_of` recupera o índice posicional e o
`membership` decide o voxel. A escolha é da geometria, nunca de um re-binning próprio — é o que
mantém a célula alinhada ao que a agregação de fato persistiu.

Dois pontos de recusa, ambos explícitos:

- a referência cita um mapa que não é o mapa de origem da agregação;
- o ponto não cai em voxel nenhum (`-1`), o que acontece quando a agregação é mais estreita
  que o mapa bruto que lhe foi passado.

A contagem por `(voxel, entidade)` vira `EntitySupport`, ordenada por entidade.

## 3. Escrita

O writer recebe o suporte, a política e a linhagem já montada. Ele calcula as três figuras do
run — células, entidades distintas e peso total — e grava, dentro de um `AtomicRunDirectory`:

```text
README.md              # texto humano
manifest.json          # metadata autoritativa + inventário com tamanho e SHA-256
lineage.json           # os três upstreams, verbatim
config.json            # schema, documento da política, fingerprint, grade e regra
environment.json       # interpretador e plataforma
outputs/cells.jsonl    # uma célula por linha, na ordem do suporte
outputs/support.json   # cabeçalho: mapa, frame, grid_fingerprint, aggregation_rule
metrics/support.json   # as três figuras
```

Um diretório existente nunca é substituído. Como `publish()` é quem monta o inventário, o
writer relê o manifest do disco para devolver o que de fato publicou — não é redundância, é a
única forma de o retorno carregar os hashes reais.

## 4. Leitura

O reader abre o manifest (recusando um `schema_version` que não entende), lê o cabeçalho de
`outputs/support.json` e decodifica cada linha de `cells.jsonl` **pelos construtores de
domínio**, então uma linha adulterada falha ao decodificar em vez de virar uma célula
inválida. `verify_integrity()` confronta primeiro o inventário com o disco, depois reconstrói
as células, entidades distintas e peso total. Compara os campos derivados do manifest com o
suporte e confere os valores repetidos em `outputs/support.json`, `config.json` e
`metrics/support.json`. Assim uma mudança só no manifest também é detectada.

## Caminhos paralelos

### Proveniência

| O quê | Onde vive |
| --- | --- |
| ContextMap de origem | `manifest.source_context_map_id` e `lineage.json` (id, schema e `content_identity`) |
| Mapa bruto | `manifest.source_map_id`; em `lineage.json`, o `run_id`, o `map_id`, o `configuration_fingerprint` e o digest de conteúdo contratual |
| Agregação voxel | `lineage.json`: `run_id`, `policy_fingerprint`, `aggregation_rule` e o digest de conteúdo contratual |
| Grade | `manifest.grid_fingerprint`, e dentro de **todo** `SpatialCellId` |
| Regra de junção | `manifest.policy_fingerprint` e o documento completo em `config.json` |
| Código | `manifest.code_version` e `environment.json` |

A linhagem é gravada verbatim e o writer não a interpreta: ele carrega a forma que
`derivation.py` montou a partir dos três manifests estrangeiros.
Os digests de conteúdo do mapa bruto e da agregação usam `artifact_digest`: identidade do run,
schema e hashes dos arquivos contratuais do próprio manifest. O `content_identity` do
ContextMap vem do manifest validado pelo leitor dele, depois da verificação dos hashes do
payload.

### Identidades

`SpatialCellId` é `spatial-cell:<grid_fingerprint>:<kx>_<ky>_<kz>` — função pura da grade e da
chave, sem registro de identidade e sem nenhum índice de ponto. `SpatialSupportRunId` vem do
chamador. O `aggregate_index` **não** é identidade: é o índice de transporte sob o qual a
geometria de apresentação serve o centroide daquele voxel, e é por ele que o viewer alinha uma
linha de suporte ao ponto renderizado.

### Unidades e frames

`centroid_m` e `bounds` estão em metros, no frame da grade, e `SemanticSpatialSupport` recusa
qualquer célula cujo `bounds` declare outro frame. `point_count` conta medições brutas do
voxel; `support_weight` conta pontos de uma entidade naquele voxel. Nenhum dos dois é
probabilidade, score ou peso de fusão.

### Configuração

Não há. A capability não lê YAML, env nem parâmetro de runtime: o chamador fornece quatro
caminhos, um `run_id` e um `code_version` opcional. A grade e a regra de agregação chegam
**pelo conteúdo** da agregação, não por configuração.

## As duas ordenações

As células saem ordenadas por índice de agregado, e isso é ordem de chave de voxel porque
`VoxelAggregation` garante agregados ordenados por `(x, y, z)`. A garantia é de
`geometric_mapping`; aqui ela é apenas consumida, e um teste a fixa.

Dentro de uma célula, o suporte é ordenado por `entity_id` e é único — esse invariante é desta
capability e é imposto na construção de `SpatialCell`.

## O que não existe neste fluxo

- nenhum estágio de runtime: a produção é o comando `contextmap derive-spatial-support`;
- nenhum downstream dentro do repositório: o consumidor é o viewer, fora dele;
- nenhuma escrita nos três artifacts de origem — eles são só lidos;
- nenhuma hipótese semântica por célula, nenhum embedding, nenhum `observation_count` por
  `(célula, entidade)`.
