# Arquitetura de Point Representation

Este documento declara o que a capability possui, em que direção suas dependências apontam, e quais
regras um review pode verificar no código. Para a responsabilidade de cada arquivo, ver
[`code-map.md`](code-map.md); para o caminho dos dados, [`data-flow.md`](data-flow.md).

## Ownership

A capability possui seis conceitos, e é a única fonte de verdade sobre eles:

| Conceito | Significa | Arquivo dono |
| --- | --- | --- |
| suporte local | quais elementos de geometria formaram o insumo de um vetor, sob qual política, e como suas coordenadas foram preparadas | `support.py`, contratos em `models.py` |
| representação | o vetor ancorado a uma geometria, com forma, dtype, normalização, referência ao payload e proveniência | `models.py` |
| identidade do espaço | o que torna dois vetores **comparáveis** | `comparability.py` |
| seleção de centros | qual geometria um run representa, como política versionada | `center_selection.py` |
| execução | resolver centro → suporte → vetor com um encoder, e falhar de forma explícita | `service.py` |
| run artifact | o run persistido, imutável, auto-descritivo | `run_artifact/` |

A capability **não** possui, e nunca deve passar a possuir:

- label, `SemanticClaim`, embedding visual ou qualquer campo semântico — `visual_perception`;
- fusão entre observações — `semantic_fusion`;
- identidade e resolução de entidade — `semantic_mapping`, `entity_resolution`;
- a geometria persistente e seu índice espacial — `geometric_mapping` (esta capability apenas **lê**,
  pela porta `GeometrySource`);
- projeção 2D↔3D, visibilidade, oclusão — `sensor_association`;
- **qual encoder roda** — `runtime`, no composition root.

## Direção de dependências

```
runtime (composition root)
    │  escolhe o encoder, injeta o PTv3Runtime, conduz o estágio
    ↓
point_representation  ← contratos e regras científicas
    ↑
    │  implementa a porta PointEncoder
backends/geometric_descriptor.py        backends/ptv3/
                                             ↑
                                             │  implementa a porta PTv3Runtime
                                        Pointcept / torch / CUDA (fora do repositório)
```

A regra: **detalhe externo depende de contrato interno, nunca o contrário.** Um encoder implementa
`PointEncoder` definido em `ports.py`; nenhum contrato público desta capability menciona torch,
numpy, Pointcept, device, checkpoint ou tensor. `models.py`, `ports.py`, `comparability.py`,
`serialization.py`, `support.py`, `service.py` e `center_selection.py` são stdlib puro.

### Dependências permitidas

| De | Para | Por quê |
| --- | --- | --- |
| toda a capability | `contextmap.shared` | `Vector3`, `AtomicRunDirectory`, `FileEntry`, `DEBUG_DIRECTORY`, `check_file_inventory` |
| `support.py`, `service.py`, `serialization.py`, `run_artifact/`, `center_selection.py` | `contextmap.geometric_mapping` | `GeometrySource`, `GeometryPoint`, `GeometryReference`, `GeometryId`, `MapId`, `Bounds3D`, `GeometricMap` |
| **só** `center_selection.py` | `contextmap.sensor_association` | `SpatialObservation`: a política ancora uma representação em cada observação espacial |

Todo import cross-capability passa pela raiz pública (`from contextmap.geometric_mapping import …`).
Não existe nenhum import de caminho interno de outra capability, e isso é verificado por
`tests/architecture/test_boundaries.py`.

### Dependências proibidas

- qualquer coisa sob `contextmap.semantic_fusion`, `semantic_mapping`, `entity_resolution`,
  `visual_perception`, `artifact`, `evaluation`, `runtime` — todas são consumidoras desta, e o
  inverso criaria ciclo;
- torch, numpy, spconv, torch_scatter, timm, Pointcept em qualquer módulo fora de
  `backends/ptv3/` — e, dentro dele, apenas de forma diferida (ver abaixo);
- filesystem fora de `run_artifact/` e de `backends/ptv3/checkpoint.py`;
- leitura de YAML, env ou parâmetro de ROS em qualquer lugar: configuração chega **tipada**, por
  construtor.

## API pública

`src/contextmap/point_representation/__init__.py` é a única superfície cross-capability: 38 nomes com
`__all__` explícito. **Nenhum backend é exportado.** `GeometricDescriptorEncoder`, `PTv3Config`,
`PTv3PointEncoder` e `PointceptPTv3Runtime` existem apenas nos seus próprios caminhos de módulo.

Consumidores reais hoje:

| Consumidor | O que usa |
| --- | --- |
| `runtime/executors.py` | a política de centros, o serviço, o writer, a porta do encoder — conduz o estágio |
| `runtime/composition.py` | `backends/geometric_descriptor` e `backends/ptv3`, por caminho de módulo, dentro das factories |
| `entity_resolution/representation_comparison.py` | `PointRepresentationRunReader` e `ensure_compatible_representations` — o único consumidor que **executa** o artifact |
| `semantic_fusion`, `semantic_mapping` | só `PointRepresentationId` e `PointRepresentationRunId`, como tipo de referência e de linhagem |
| `evaluation/point_representation.py` | o harness de ablação |

## Adapters e backends

`backends/__init__.py` **não exporta nada** e não deve passar a exportar. Um backend é alcançado pelo
próprio caminho de módulo, e só do composition root — que é exatamente o que
`tests/architecture/test_runtime_boundaries.py` exige, e por isso os imports em `composition.py` são
locais à factory que os usa. Adicionar uma fachada aqui criaria uma segunda superfície pública e
importaria eagerly todo backend futuro, destruindo a garantia de peso de import zero.

### `geometric_descriptor.py` — a linha de base determinística

Descritor de 14 features de covariância local, aritmética IEEE-754 em Python puro, sem nenhuma
dependência de modelo. É o **controle** científico: qualquer encoder aprendido precisa demonstrar
vantagem sobre ele por avaliação, não por argumento.

### `backends/ptv3/` — o encoder aprendido opcional

Um subpacote por backend, com os concerns separados por motivo de falha:

| Módulo | Falha quando |
| --- | --- |
| `adapter.py` | a configuração é inválida, ou o runtime injetado viola o contrato |
| `pointcept_backbone.py` | a configuração não é honrável pela arquitetura do checkpoint adotado |
| `voxelization.py` | o suporte não é mapeável para a entrada esparsa do backbone |
| `checkpoint.py` | os bytes do checkpoint não são os fixados, ou não contêm pesos de backbone |
| `pointcept_module.py` | uma biblioteca opcional ou o checkout do Pointcept não está presente |
| `pointcept_runtime.py` | o device falha: sem CUDA, sem memória, pesos não carregam |

`backends/ptv3/__init__.py` re-exporta **só** a superfície do adapter, e importa só `adapter.py`, que é
stdlib puro. Consequência verificável: importar o backend não carrega numpy, torch, spconv,
torch_scatter nem Pointcept — asserção explícita em `test_ptv3_backend.py` e
`test_ptv3_pointcept_runtime.py`.

`pointcept_module.py` é o único módulo autorizado a importar biblioteca de modelo opcional ou a
escrever em `sys.modules`. Esse side effect de processo existe para contornar os imports de treino do
`__init__` do Pointcept, é idempotente, e está declarado na docstring do módulo — não escondido no
meio de uma classe de runtime.

`PointceptPTv3Runtime` **só é alcançável por um provider externo ao repositório**
(`resources.providers["point_representation.encoder"]` aponta para um módulo seu, por exemplo
`ablation_providers:load_ptv3`). Isso é design, não código morto: o runtime pesado é injetado por
inversão de dependência, e o repositório não carrega torch nem CUDA por causa dele.

## Composição no runtime

O estágio é **opcional** e desligado em `canonical/1`. Com
`pipeline.stages.point_representation = true`, o runtime escolhe dois pontos de variação —
`point_representation.center_selection` e `point_representation.encoder` — e compõe o executor.

Duas regras do composition root que esta capability depende para ser cientificamente defensável:

- **não existe fallback entre encoders.** Se o `ptv3` é selecionado e seu `PTv3Runtime` não é
  fornecido, o estágio fica de fora e é reportado como tal. Ele nunca é silenciosamente substituído
  pelo descritor determinístico, porque isso produziria um run cujo manifest não descreve o que
  realmente rodou;
- **não existe estágio sem identidade de código.** Sem `code_version`, o estágio fica de fora: um run
  que não sabe qual código o produziu não é auditável.

Dentro da capability a mesma regra vale um nível abaixo: `RepresentationService` não troca de encoder
quando o selecionado falha. Um suporte que o encoder declara inencodável, ou cujo resultado não é
finito, vira `FailedSupport` registrado; nada é substituído por vetor zero ou default.

## Invariantes arquiteturais

1. **Um run artifact abre com a biblioteca padrão.** Nenhum caminho de leitura importa numpy, torch
   ou a geometria. Por isso `center_selection_id()` vive em `run_artifact/manifest.py` e não em
   `center_selection.py`, que importa `sensor_association` — ler um run não pode arrastar a
   capability de associação.
2. **Dimensionalidade igual nunca torna dois vetores comparáveis.** Só o mesmo fingerprint de
   `RepresentationSpace` torna. A política de suporte faz parte do espaço: o mesmo encoder sob raio
   0,25 m e sob 1 m produz vetores incomparáveis.
3. **O suporte faz parte da identidade.** Um vetor nunca é computado de um XYZ único; sempre de um
   `PointSupport` que lista exatamente quais `GeometryReference` formaram o insumo.
4. **A ordem do suporte é contrato.** Centro primeiro, depois distância euclidiana crescente ao
   centro, empate por `geometry_id` crescente — independente da ordem em que a fonte devolve os
   pontos.
5. **Nenhum canal é concatenado.** `PointRepresentation` nunca carrega label, claim, embedding visual
   ou feature de outro canal num vetor opaco.
6. **`outputs/` e `metrics/` são contratuais; `__DEBUG__/` nunca é.** Remover `__DEBUG__/` não pode
   invalidar um run, e nenhum consumidor a jusante pode depender dele.
7. **Escrita é atômica.** Um write interrompido nunca se parece com um run terminado, e um run
   terminado nunca é modificado. Re-execução gera outra identidade.
8. **Escores heterogêneos não são misturados.** Esta capability não produz confidence; produz um
   vetor e a identidade do espaço dele. Comparar dois vetores é responsabilidade de quem consome, sob
   a regra de comparabilidade acima.

## Dívida arquitetural declarada

- **O run artifact não tem consumidor a jusante.** A fusão semântica não tem entrada `representation`
  e a configuração do runtime recusa esse canal de evidência mesmo com o estágio ligado. O run é
  saída derivada. Levar essa evidência até o `ContextMap` exige um contrato novo de consumo na fusão
  e na Entity Resolution — pergunta de pesquisa própria, não dívida de implementação.
- **`metrics/norms.json` grava `"non_finite_values": 0` como literal**, não um valor contado. Vetores
  não finitos são recusados antes de chegar ao writer, então o campo é estruturalmente sempre zero.
  Remover é mudança de schema; é uma decisão de contrato pendente, não um bug.
- **`POINTCEPT_RUNTIME_VERSION` é definido e nunca referenciado.** Ele versiona o mapeamento
  suporte→entrada→saída, mas não entra em nenhuma identidade nem fingerprint, ao contrário de
  `DESCRIPTOR_VERSION` e `PTV3_ADAPTER_VERSION`. Estampá-lo mudaria identidade de encoder; está
  pendente de decisão.
- **O writer acumula o run inteiro em memória** (payload em `bytearray`, registros em lista) até
  `finalize`. Adequado à escala atual, uma âncora por observação espacial; não a runs do tamanho do
  mapa.
- **`__DEBUG__/` é escrito pelo writer, sem dono separado.** Um `run_artifact/debug.py` seria uma
  classe com um único consumidor, criada por simetria e não por necessidade demonstrada. Se o volume
  de diagnóstico crescer, é aí que ele deve nascer.
