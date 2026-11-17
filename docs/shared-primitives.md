# Primitivas compartilhadas e tipos transversais

Este documento define quando um conceito pode pertencer ao namespace `contextmap.shared` e quando deve permanecer sob ownership de uma capability.

A regra principal é:

> `shared` existe somente para primitivas sem owner semântico claro que atravessam múltiplas capabilities. Ele não é um local de conveniência para evitar dependências ou ciclos.

As regras gerais de ownership e imports permanecem em [architecture.md](architecture.md) e [module-api.md](module-api.md).

## Estado atual de `contextmap.shared`

`shared` continua deliberadamente mínimo. As primitivas transversais materializadas são `SourceTimestamp` (com `to_record()`/`from_record()`, o formato de registro compartilhado do timestamp e do clock) e as primitivas geométricas de `contextmap.shared.geometry` (`Vector3`, `Quaternion`, `RotationMatrix`, validação, norma e normalização de quaternions, `quaternion_multiply`, `quaternion_conjugate`, `rotate_vector`, `quaternion_to_rotation_matrix`, `quaternion_angle_between`, `compose_rigid` e `invert_rigid`). `contextmap.shared.run_directory` materializa a mecânica transversal de diretório de run descrita abaixo, e `contextmap.shared.provenance` materializa `BackendProvenance`, a identidade mínima do backend que produziu uma evidência (justificativa em [Provenance genérica](#provenance-genérica)); os demais conceitos permanecem com seus owners de domínio enquanto não houver necessidade real de compartilhamento.

```mermaid
flowchart LR
    SH["contextmap.shared<br/>SourceTimestamp + geometry"] --> ING["contextmap.ingestion"]
    SH --> SE["contextmap.state_estimation"]
    ING --> SO["SourceObservation"]
    SO --> VP["contextmap.visual_perception"]
    VP --> E["Region2D / VisualFeature /<br/>SemanticClaim / SemanticScore"]
    E -. não mover para shared .-> DOWN["capabilities downstream"]
```

`FrameId`, `RigidTransform`, calibração e IDs de observação continuam hoje sob ownership de Ingestion. O fato de futuros módulos também precisarem de frames ou transforms não autoriza mover esses tipos para `shared` antes de existir um contrato transversal real.

### Primitivas geométricas em `shared.geometry`

Os aliases `Vector3` e `Quaternion` e as funções sobre quaternions entraram em `shared` porque atendem aos critérios abaixo:

- **quem usa:** `state_estimation` (validação de `PoseEstimate.orientation`, frame graph e preflight, que compõem e invertem transforms), com `geometric_mapping` e `sensor_association` consumindo a mesma álgebra para compor `T_map_body(t)` com extrínsecos estáticos;
- **mesma semântica:** quaternion `(x, y, z, w)`, unitário, sem reordenação em relação a `RigidTransform` e `ExternalPoseMeasurement` de Ingestion;
- **owner natural:** nenhum; a álgebra de quaternions não pertence a uma capability de domínio;
- **API pública:** apenas tuplas, sem NumPy ou biblioteca de robótica.

`normalize_quaternion` devolve um quaternion unitário ou recusa com `ValueError`, nunca um resultado que só pareça rotação: recusa componentes não finitos e toda norma fora dos floats normais e finitos, ou seja, nula, subnormal (abaixo de `sys.float_info.min`, onde os componentes já perderam a precisão de uma direção e a divisão não dá norma 1) ou que transborda o float64 (a divisão daria zeros) (#647). Quem normaliza na fronteira de uma capability traduz a recusa no erro semântico dela: `invalid_orientation` na validação de amostras de pose, `FrameGraphError` no frame graph, `StateEstimationEvaluationError` na leitura de uma referência TUM.

`shared.geometry` cresce apenas quando uma issue tem consumidor real da nova função; tipos de domínio (`PoseEstimate`, `GeometryPoint`, `RigidTransform`) permanecem com seus owners.

### Mecânica de run directory em `shared.run_directory`

`docs/ARTIFACTS.md` fixa as mesmas regras para todo run artifact: uma escrita interrompida não pode parecer um run finalizado, um run finalizado é imutável e nunca sobrescrito, o manifest inventaria cada arquivo contratual com tamanho e hash. `AtomicRunDirectory`, `FileEntry`, `check_file_inventory` e `publish_run_directory` implementam essas regras uma vez, sem conhecer o conteúdo de um run. Um payload maior que a memória é gravado em fluxo por `AtomicRunDirectory.open_binary`, que hasheia durante a escrita, e `check_file_inventory` hasheia em blocos; ambos surgiram do primeiro consumidor real (o payload de geometria) e não mudam as regras. `AtomicRunDirectory.inventory()` devolve, antes de publicar, o inventário que `publish` vai gravar, com o tamanho e o hash contados na escrita; surgiu do `ContextMapArtifact`, cuja identidade de conteúdo cobre o inventário e cujas tabelas são gravadas em fluxo (#600). A publicação é também durável (#616): `write_bytes` e `open_binary` sincronizam cada arquivo em disco (`fsync`) ao escrevê-lo, e `publish` sincroniza o manifest, o README e cada diretório do run antes do `rename` e o diretório pai depois dele, então uma queda de energia não deixa um run publicado com arquivos vazios ou truncados; a sequência e o custo estão em [ARTIFACTS.md](ARTIFACTS.md#durabilidade). `hash_file` mede um arquivo em disco como uma entrada de inventário o registra (tamanho e `sha256:<hex>`), lendo em blocos: é a mesma função que `check_file_inventory` usa, e o feature store de Visual Perception a usa para conferir um payload na leitura, em vez de manter uma cópia própria do hash em blocos (#645). A publicação nunca substitui o caminho final (#637): o `rename` é feito com `renameat2(..., RENAME_NOREPLACE)`, que o kernel recusa se qualquer coisa existir no destino, inclusive um diretório vazio criado por outro processo depois da checagem de `__init__`; um `rename(2)` comum substituiria esse diretório em silêncio. A chamada só existe no Linux (kernel 3.15, glibc 2.28) e em sistemas de arquivos que implementam a flag; fora deles `publish` levanta `RunDirectoryError` e não publica, sem recorrer a um `rename` que possa substituir o destino (detalhes em [ARTIFACTS.md](ARTIFACTS.md#publicação-sem-substituição)). Uma escrita que falha, em `write_bytes` ou em um fluxo de `open_binary` (disco cheio, `EIO`, `fsync` recusado ou exceção dentro do bloco), tem o mesmo contrato nos dois métodos (#639): o arquivo parcial é removido do diretório temporário, o erro original chega ao chamador (uma falha ao remover vira só uma nota nele) e o run deixa de ser publicável, mesmo que escritas posteriores deem certo; assim um chamador que captura o erro e segue nunca publica um arquivo truncado fora do inventário. `is_run_relative_path` é a regra única de caminho dentro de um run (relativo, sem `..`, não vazio): a escrita a exige e `check_file_inventory` reporta, sem abrir, uma entrada de manifest fora do run, de modo que um manifest adulterado nunca faz um leitor ler outro arquivo do sistema.

- **quem usa:** `ingestion`, `state_estimation`, `geometric_mapping`, `sensor_association`, `point_representation`, `semantic_fusion`, `semantic_mapping`, `entity_resolution`, `spatial_relations` e `artifact` com `AtomicRunDirectory`, e `visual_perception` com `publish_run_directory`, todos com a mesma regra de `ARTIFACTS.md`;
- **owner natural:** nenhuma capability de domínio; a regra é global;
- **API pública:** apenas `pathlib` e tipos primitivos, sem NumPy nem SDK;
- **o que continua com cada capability:** quais arquivos existem, os campos do manifest e o que torna um run válido.

`publish_run_directory` aplica a mesma publicação (sincronização, `rename` sem substituição, sincronização do pai) a um diretório temporário que o owner montou com writers próprios. Surgiu do `PerceptionRunWriter`, cujos stores de máscara e de feature, auditoria semântica e diagnósticos de feature gravam num diretório e não passam por `write_bytes` (#640); cada arquivo é sincronizado na publicação, e a checagem do inventário continua com o owner. O `SequenceArtifactWriter` de Ingestion usa o próprio `AtomicRunDirectory` (#622), com três ajustes sem layout embutido: `publish(readme=None)` para um layout sem `README.md`, `refuse_existing_early=False` para quem documenta a colisão só na publicação (que a recusa do mesmo jeito) e `discard()`, que remove o temporário na hora e propaga uma falha de remoção, para quem mantém o run aberto entre chamadas e precisa saber que a limpeza não aconteceu.

## Critérios para entrar em `shared`

Um tipo só pode ser movido para `shared` quando todos os critérios abaixo forem verdadeiros:

1. é usado por mais de uma capability com **a mesma semântica**;
2. nenhuma capability possui responsabilidade de domínio claramente superior sobre o conceito;
3. o tipo é pequeno, estável e independente de backend;
4. colocá-lo em `shared` reduz duplicação conceitual real, não apenas linhas parecidas;
5. ele não existe para quebrar artificialmente um ciclo entre módulos;
6. sua API pública não exige ROS, NumPy, Torch, SDK de modelo ou outro runtime pesado.

Se houver dúvida sobre ownership, o conceito permanece no módulo que o introduz até existir evidência de uso transversal real.

## Superfície inicial permitida

A arquitetura do canonical pipeline reconhece as seguintes famílias como candidatas legítimas a `shared` quando a implementação passar a precisar delas.

### Tempo

Uma representação temporal genérica pode ser compartilhada porque timestamps aparecem em observações, poses, geometria, associação e lineage.

O contrato precisa preservar a semântica temporal necessária para evitar comparação cega entre relógios diferentes. Conceitualmente:

```text
Timestamp
├── value
├── unit
└── clock_id / time_domain, quando necessário
```

Não assumir que dois números de timestamp pertencem ao mesmo clock.

A conversão de timestamps ROS/dataset para o tipo canônico pertence aos adapters de Ingestion.

### Frames e transformações 3D

As seguintes primitivas podem ser compartilhadas porque possuem significado geométrico transversal:

```text
Vector3
Quaternion
Transform3D
CoordinateFrame / FrameId
```

O contrato deve declarar convenções necessárias, principalmente ordem de quaternion e direção de transformações.

Preferência de notação:

```text
T_A_B
```

representa uma transformação que leva coordenadas expressas em `B` para `A`.

Tipos de domínio mais ricos, como `PoseEstimate`, `GeometryPoint` ou `CameraIntrinsics`, continuam sob ownership das capabilities correspondentes.

### Identificadores genéricos

Um identificador só pertence a `shared` quando sua semântica é realmente transversal.

É aceitável compartilhar uma primitive/base leve para representação de IDs ou referências, mas IDs de domínio continuam module-owned:

```text
SourceObservationId   -> ingestion
RegionPerceptionRunId -> region_perception
VisualEvidenceRunId   -> visual_perception
SemanticInterpretationRunId -> semantic_interpretation
GeometryReference     -> geometric_mapping
EntityId              -> semantic_mapping
RelationId            -> spatial_relations
ArtifactId            -> artifact, se possuir semântica própria de artifact
```

Não criar uma hierarquia complexa de IDs apenas para evitar strings.

### Provenance genérica

A arquitetura precisa de um vocabulário transversal mínimo para referências de lineage, como identidade do produtor, código/configuração e referências upstream.

Esse núcleo pode viver em `shared` se múltiplas capabilities realmente o reutilizarem com a mesma semântica.

`BackendProvenance` é hoje esse núcleo, em `contextmap.shared.provenance`, e entrou pelos critérios acima:

- **quem usa:** `region_perception`, `visual_features`, `semantic_interpretation`, `semantic_scoring`, `semantic_fusion`, `semantic_mapping`, `sensor_association`, `evaluation` e `runtime`, todas anexando o registro à sua própria evidência;
- **mesma semântica:** nenhuma capability redefine os campos nem o significado de `capability`/`configuration_fingerprint`;
- **owner natural:** nenhum; backends de todas as capabilities produzem o registro, e mantê-lo em `visual_perception` fazia oito capabilities dependerem da nona só para nomear o backend que executou;
- **duplicação real:** o tipo estava definido duas vezes (`visual_perception/models.py` e `visual_perception/backend_provenance.py`), com campos idênticos mas classes distintas em runtime, de modo que `isinstance` e `==` falhavam entre caminhos de import;
- **API pública:** seis campos `str`, nenhum runtime pesado.

A entrada não serve para quebrar ciclo: não havia ciclo envolvendo o tipo, apenas direção invertida. Provenance específica de capability — `SemanticInferenceProvenance`, `RegionProvenance`, `FeatureResolutionEnhancementProvenance` — continua com seu owner, que é exatamente a forma prescrita abaixo.

Provenance específica continua module-owned. Exemplo:

```text
common producer/config/code identity
    +
Perception provenance específica
```

não significa colocar prompts, checkpoints, calibration details e campos de cada capability dentro de um único objeto global de provenance.

## Conceitos explicitamente fora de `shared`

Os seguintes conceitos possuem owner de domínio e não devem ser movidos para `shared` por conveniência:

```text
SourceObservation
CanonicalSequence
CameraIntrinsics
ExternalPoseMeasurement
PoseEstimate
Trajectory
PreparedImage
Region2D
VisualFeature
EmbeddingSpace
SemanticClaim
SemanticScore
SceneContext
GeometryPoint
GeometryReference
GeometricMap
SpatialObservation
ObservationQuality
PointRepresentation
RepresentationSpace
FusionSupport
FusedEvidence
Entity
ResolvedEntity
Relation
ContextMap
ContextMapArtifact
```

O fato de vários módulos consumirem esses tipos não remove o ownership do produtor.

## `Confidence` não é uma primitive compartilhada

Não criar um `shared.Confidence` genérico no canonical pipeline.

Os valores que parecem “confidence” possuem semânticas distintas:

```text
VLM confidence
semantic scorer similarity
observation quality
fusion contribution weight
entity uncertainty
relation support
```

Uma abstraction comum facilitaria combinações semanticamente incorretas.

Cada módulo deve nomear e documentar seu sinal conforme a semântica real. Conversão para probabilidade só é válida quando houver uma calibração explicitamente definida.

## Calibration ownership

Calibration não entra em `shared` apenas porque várias etapas a consomem.

Ownership definido:

```text
ingestion
    normaliza e possui dados canônicos de calibration
    ├── camera intrinsics/model/distortion
    ├── static extrinsics
    └── frame metadata

sensor_association
    possui aplicação da calibration na projeção 3D -> imagem
```

`state_estimation` pode consumir static extrinsics/frame metadata necessários ao frame graph, mas não passa a possuir calibration.

Adapters podem converter formatos específicos de dataset/ROS para o contrato público de Ingestion. Downstream nunca deve parsear YAMLs ou mensagens ROS de calibration diretamente.

## NumPy e bibliotecas numéricas

Primitivas públicas compartilhadas devem ser **library-neutral**.

Elas podem oferecer conversão explícita para representações numéricas quando necessário, mas não devem exigir `numpy.ndarray` como identidade do contrato público.

Motivos:

- evita carregar NumPy apenas para interpretar metadata simples;
- impede shape/dtype implícitos como parte não documentada do contrato;
- facilita serialização e testes;
- mantém Torch/JAX/CUDA restritos à infraestrutura/backends que realmente precisam deles.

Cálculos internos podem usar NumPy livremente dentro da capability adequada. A regra vale para a fronteira pública, não para toda implementação.

NumPy é **dependência base** do pacote (fixado abaixo de 2.4 por causa dos stubs; ver `pyproject.toml`): a acumulação de Geometric Mapping e as buscas de vizinhos da avaliação precisam dele. Isso não afrouxa a regra acima. Continuam valendo, e são testadas: nenhuma primitiva pública exige `numpy.ndarray`, e os leitores de artifact abrem um run sem importar NumPy, porque o import é tardio e fica dentro do cálculo que realmente o usa.

## Estrutura materializada

A árvore atual contém somente os módulos com consumidores reais:

```text
src/contextmap/shared/
├── __init__.py
├── time.py               # SourceTimestamp
├── geometry.py           # Vector3, Quaternion, RotationMatrix e operações rígidas
├── provenance.py         # BackendProvenance
├── payload_reference.py  # ArtifactReference, JsonScalar
└── run_directory.py      # escrita atômica, inventário e integridade
```

`identifiers.py` não existe; só deve ser criado se surgir um contrato transversal real.

`shared/__init__.py` deve seguir a mesma política de API pública das demais capabilities: exports explícitos e pequenos.

### Admissões registradas

Três símbolos entraram em `shared` durante a decomposição de `visual_perception`.
`BackendProvenance` está justificado em [Provenance genérica](#provenance-genérica); os dois
abaixo vieram depois, e as respostas do [critério de revisão](#critério-de-revisão) ficam aqui,
porque é delas que a admissão depende:

**`ArtifactReference`** (`payload_reference.py`) — referência a um payload persistido, com
caminho e hash. Usado por `visual_perception` (`PreparedImage.payload_artifact`,
`TransformationRecord.output_image`), `region_perception` (`RegionCandidate.mask_reference`,
mask store, grounding), `sensor_association` e o `runtime`. É o irmão em contrato do
`FileEntry` de `run_directory.py`, que já vive aqui e mede o mesmo par caminho+hash para o
inventário de um run, e [ARTIFACTS.md](ARTIFACTS.md) declara a regra de referência preguiçosa
com hash para **todo** artifact — então nenhuma capability é owner superior. Colocá-lo em
`region_perception` faria `PreparedImage` depender daquela capability por um conceito que não é
de região. Nenhuma biblioteca na API pública.

**`JsonScalar`** (`payload_reference.py`) — alias estrutural de um escalar JSON. Usado por
quatro módulos de `region_perception`, oito de `semantic_interpretation` e por
`TransformationRecord.parameters`. Mesma semântica em todos, nenhum owner, nenhum runtime.

Nenhum dos três entrou aqui para quebrar um ciclo. Os três estavam em módulos cujo nome dizia
outra coisa — `visual_perception/region_models.py` para os dois últimos, que não eram nem
contrato de região nem de percepção —, e era essa a inversão de ownership. Ver a
[regra para ciclos](#regra-para-ciclos) abaixo, que continua valendo.

## Regra para ciclos

Um ciclo como:

```text
module_a -> module_b -> module_a
```

não pode ser “resolvido” automaticamente movendo tipos para `shared`.

A análise deve primeiro verificar:

1. ownership incorreto;
2. direção de dependência errada;
3. necessidade de um port no consumer;
4. responsabilidade que deveria estar em runtime/composition;
5. artifact/contract público que já pode quebrar a dependência.

Somente depois disso uma primitive genuinamente transversal pode ser considerada para `shared`.

## Exemplos

### Correto

`Vector3` é usado com a mesma semântica por state estimation e geometric mapping e não carrega regras específicas de nenhuma delas. Pode ser `shared`.

### Incorreto

Mover `GeometryReference` para `shared` porque Sensor Association também o consome. O owner permanece Geometric Mapping; Sensor Association depende da API pública de Geometric Mapping.

### Incorreto

Criar `shared.SemanticEvidence` para unificar `SemanticClaim`, `SemanticScore`, `FusedEvidence` e relation evidence. Esses conceitos possuem semânticas e owners diferentes.

## Critério de revisão

Todo novo símbolo proposto para `shared` deve responder no PR:

- quais capabilities o utilizam hoje;
- por que a semântica é exatamente a mesma entre elas;
- por que nenhum módulo é owner natural;
- quais dependências ele evita sem mascarar um problema arquitetural;
- quais bibliotecas aparecem em sua API pública.

Sem respostas concretas, o símbolo permanece module-owned.
