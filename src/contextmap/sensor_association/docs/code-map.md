# Mapa do código de Sensor Association

Este documento responde "qual arquivo devo abrir para entender ou alterar esta
responsabilidade?". Ele descreve **ownership**, não apenas o conteúdo atual. Se uma função
nova não encaixar na responsabilidade declarada do arquivo, a primeira pergunta não é onde
colocá-la, e sim de quem ela é.

## Raiz da capability

| Arquivo | Responsabilidade | Entrada -> saída | Não deve possuir |
| --- | --- | --- | --- |
| `__init__.py` | fachada pública cross-capability | símbolos internos -> API estável | implementação; passo intermediário exportado |
| `models.py` | contratos canônicos de evidência (`SpatialObservation` e cia.) | valores validados -> contrato | NumPy, ROS, SDK; regra de projeção ou de pertencimento |
| `serialization.py` | contratos <-> registros de primitivas JSON | contrato <-> registro, com revalidação | layout de arquivo; decisão de quando gravar |
| `errors.py` | o erro de entradas incompatíveis da capability | — | erro de I/O de artifact (é de `artifacts/`) |
| `run_definitions.py` | as versões de definições que identificam um run | constantes dos módulos donos -> mapa único | a definição em si (cada uma vive com seu dono) |
| `quality.py` | contrato `ObservationQuality` e seus componentes | medidas -> contrato tipado | derivação; confiança semântica; escalar combinado |
| `quality_derivation.py` | derivar a qualidade de cada observação de um frame | pertencimento + observações -> qualidades | definição de componente (é de `quality.py`) |
| `dense_sampling.py` | amostrar um mapa denso sobre a geometria visível | visibilidade + `DenseFeatureMap` -> índices e pesos | vetor de feature copiado; fusão entre frames |
| `service.py` | validar lineage do request/frame e compor a execução streaming | `SensorAssociationRequest` -> `SensorAssociationOutcome` + frames no sink | regra científica própria; escolha de backend; I/O |

## `camera/` — projeção de câmera calibrada

| Arquivo | Responsabilidade | Entrada -> saída | Não deve possuir |
| --- | --- | --- | --- |
| `__init__.py` | fachada do subpacote | símbolos internos -> port + factory | implementação de modelo |
| `projection.py` | port `CameraProjection`, contratos e base compartilhada | pontos no frame óptico -> `PixelProjection` | matemática específica de um modelo |
| `distortion.py` | numérica de domínio de visão e distorção radial-tangencial | plano normalizado e ângulos -> limites, inversas, jacobianos | calibração, identidade de câmera, pixel de imagem |
| `pinhole.py` | modelo pinhole (plumb_bob e racional) | ponto -> pixel, pixel -> raio | leitura de calibração; escolha de modelo |
| `fisheye.py` | modelo equidistante Kannala-Brandt | idem | idem |
| `mei.py` | modelo omnidirecional unificado (CamOdoCal) | idem | idem |
| `selection.py` | escolher e validar o modelo que a calibração declara | `CalibrationEntry` -> `CameraProjection` | matemática de projeção; conhecimento de dataset |

## `projection_chain/` — da geometria do mapa ao pixel preparado

| Arquivo | Responsabilidade | Entrada -> saída | Não deve possuir |
| --- | --- | --- | --- |
| `__init__.py` | fachada do subpacote | símbolos internos -> passos da cadeia | regra nova |
| `candidate_geometry.py` | que geometria o frame avalia, antes da projeção exata | centro óptico + política -> nuvem de candidatos + relatório | projeção; visibilidade |
| `image_transform.py` | reproduzir a cadeia de preparo da imagem registrada | `PreparedImage` -> transform pixel bruto -> preparado | reamostragem de imagem; inferência de espaço |
| `frame_projection.py` | compor pose, extrínseca e modelo e posicionar os pontos | observação + imagem preparada -> `FrameProjection` \| `RejectedProjection` | decisão de oclusão; escolha de região; rótulo |
| `visibility.py` | oclusão por suporte local de profundidade conservador | `FrameProjection` + política -> `VisibilityResolution` | escolha de região; política com valor padrão |

## `membership/` — que região suporta que geometria

| Arquivo | Responsabilidade | Entrada -> saída | Não deve possuir |
| --- | --- | --- | --- |
| `__init__.py` | fachada do subpacote | símbolos internos -> regras + contratos | regra nova |
| `models.py` | o vocabulário que as duas regras compartilham | — | regra de pertencimento |
| `region_association.py` | regra de máscara e despacho por região | visibilidade + `PerceptionResult` -> `FrameMembership` | matemática de componentes 3D; montagem do contrato |
| `box_depth_component.py` | regra de caixa e rotulagem de componentes 3D | região sem máscara + política -> `RegionMembership` \| `SkipReason` | leitura de máscara; vencedor semântico |
| `spatial_observations.py` | montar os `SpatialObservation` canônicos | `FrameMembership` -> contratos públicos | regra de pertencimento; cópia de payload |

## `diagnostics/` — condições observáveis de um frame

| Arquivo | Responsabilidade | Entrada -> saída | Não deve possuir |
| --- | --- | --- | --- |
| `__init__.py` | fachada do subpacote | símbolos internos -> diagnósticos | regra nova |
| `reprojection.py` | resíduo contra referência confiável | projeção + referência -> `ReprojectionStatistics` | criação de correspondência; estimativa a partir da associação |
| `frame_report.py` | reunir o que o frame dependeu e emitir achados | visibilidade (+ pertencimento, referência, amostras) -> `FrameDiagnostics` | o projetor inteiro; otimização de calibração |
| `time_offset_sweep.py` | reprojetar em timestamps deslocados, só para diagnóstico | projetor + offsets -> um resultado por offset | alteração de timestamp ou calibração |

## `artifacts/` — o run persistido

| Arquivo | Responsabilidade | Entrada -> saída | Não deve possuir |
| --- | --- | --- | --- |
| `__init__.py` | fachada do subpacote | símbolos internos -> writer, reader, manifest | regra de escrita ou leitura |
| `run_manifest.py` | o **formato**: schema, layout, manifest e codec binário | manifest JSON <-> contrato; bytes <-> array | escrita de run; acesso por identidade |
| `run_writer.py` | escrita streaming, registros por frame e publicação atômica | `FrameAssociation` -> bytes; `Outcome` -> manifest | leitura; regra científica |
| `run_reader.py` | leitura por identidade e integridade | diretório -> observações, qualidade, associações densas | escrita; acesso a `__DEBUG__/` |
| `debug_evidence.py` | evidência de depuração não contratual | `FrameAssociation` -> CSV, JSON e PNG em `__DEBUG__/` | qualquer dado de que alguém a jusante dependa |

## Invariantes de ownership

- **Nenhum consumidor externo importa um submódulo.** A fronteira cross-capability é
  `contextmap.sensor_association`; `tests/architecture/test_boundaries.py` recusa o resto.
- **Passos não têm export individual na raiz.** O envelope público `FrameAssociation`
  expõe projeção, visibilidade, membership e amostras ao sink de execução; isso não transforma
  seus arrays no contrato canônico ou persistente de evidência.
- **Parâmetros são explícitos.** Oclusão, caixa e tolerâncias exigem seus parâmetros;
  canais densos declaram interpolação. `CandidateGeometryPolicy(max_range_m=None)` é o
  padrão deliberado de avaliar o mapa inteiro, registrado junto das outras escolhas do run.
- **Versão de definição é identidade.** Mudar o que uma contagem significa exige uma versão
  nova; `run_definitions.py` existe para que o fingerprint e o manifest nunca discordem.
- **`__DEBUG__/` não é dependência.** Nada em `outputs/` ou `metrics/` é derivado dele, e nenhum
  estágio a jusante pode lê-lo.

## Integração e rastreabilidade

O runtime constrói políticas e readers upstream, fornece um gerador de frames e abre a
transação do writer. `semantic_fusion`, `semantic_mapping`, `point_representation` e
`evaluation` consomem a API pública, sem importar estes passos internos.

As regressões de lineage pertencem a `test_association_service.py`; a existência distribuível
do pacote de persistência é protegida por `tests/packaging/test_repository_hygiene.py` e pelo
smoke da distribuição instalada. O padrão de ignore de dados de pesquisa não pode esconder
os módulos Python de `artifacts/`.
