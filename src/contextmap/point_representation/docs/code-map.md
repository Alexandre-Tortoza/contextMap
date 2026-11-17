# Mapa de código de Point Representation

Este documento responde a uma pergunta: **qual arquivo devo abrir para entender ou alterar esta
responsabilidade?**

Ele descreve **ownership, não conteúdo atual**. Serve como contrato para review: se uma função nova
não cabe na responsabilidade declarada do arquivo, isso é discussão de ownership, não detalhe de
implementação. Tamanho de arquivo é um sinal, não o critério de divisão.

Para boundaries e direção de dependências, ver [`architecture.md`](architecture.md); para o caminho
dos dados, [`data-flow.md`](data-flow.md).

## Raiz da capability

| Arquivo | Responsabilidade | Entrada → saída | Conceitos | Não deve possuir |
| --- | --- | --- | --- | --- |
| `__init__.py` (104) | fachada pública cross-capability | símbolos internos → 38 nomes com `__all__` | o contrato estável que outras capabilities veem | regra, adapter, backend, ou export "por conveniência" |
| `models.py` (622) | contratos canônicos da capability | nada (só contratos) → `PointRepresentation`, `PointSupport`, `PreparedSupport`, `RepresentationSpace`, `SupportPolicy`, `EncoderIdentity`, `RepresentationProvenance`, `FailedSupport`, enums | suporte faz parte da identidade; nenhum payload numérico aqui, só `payload_reference` | numpy, tensor, I/O, chamada de modelo, regra de fusão, label |
| `ports.py` (93) | a porta do encoder: o ponto de substituição real | `PreparedSupport` → `EncodedVector` | por que existe variação (descritor determinístico × encoder aprendido) | implementação concreta, registry, auto-construção, import de framework |
| `comparability.py` (73) | identidade do espaço e a regra de quando dois vetores podem ser comparados | `RepresentationSpace` → fingerprint `sha256:…`; par → `None` ou erro | dimensionalidade igual **não** torna comparável; a política de suporte faz parte do espaço | distância, similaridade, agregação — comparar é do consumidor |
| `center_selection.py` (94) | política versionada de **quais** geometrias um run representa | `SpatialObservation` + `GeometrySource` → centros ordenados e deduplicados | é heurística declarada, não "o centro do objeto"; único módulo que toca `sensor_association` | extração de suporte, encoding, persistência, identidade do conjunto de centros |
| `support.py` (293) | extração determinística do suporte local e preparação de coordenadas | `GeometryReference` → `PreparedSupport` | ordem é contrato; `coordinates_m` autoritativo em metros no frame do mapa | índice espacial próprio, escolha de centros, encoder, política científica de outro dono |
| `service.py` (246) | execução independente de backend, com falha explícita | centros → iterador de `EncodedRepresentation` \| `FailedSupport` + `RepresentationMetrics` | sem fallback entre encoders; nada vira vetor zero ou default | conhecimento de backend concreto, persistência, seleção de centros |
| `serialization.py` (239) | contratos ↔ registros JSON primitivos | contrato → `dict`; `Mapping` → contrato revalidado | um registro nunca embute o vetor; decodificar revalida invariantes | layout de arquivo, filesystem, schema version |

## `run_artifact/` — o run persistido

O acordo byte a byte entre writer e reader (offsets em `outputs/representations.jsonl`) tem dono
declarado em `layout.py`, em vez de ser coincidência de "mesmo arquivo".
Camada interna: `layout` ← `manifest` ← {`writer`, `reader`}; `metrics` ← `writer`.

| Arquivo | Responsabilidade | Entrada → saída | Conceitos | Não deve possuir |
| --- | --- | --- | --- | --- |
| `__init__.py` (48) | fachada interna do artifact | módulos → 8 nomes com `__all__` | o que um chamador do artifact usa | regra de escrita ou leitura |
| `layout.py` (83) | **onde** cada arquivo de um run vive, **como** seus bytes se formam, qual schema este código escreve e entende, e o vocabulário de erro | nada → constantes, `encode_document`, `encode_lines`, os dois erros, `PointRepresentationDebugLevel` | `outputs/` e `metrics/` contratuais; `__DEBUG__/` nunca | leitura, escrita, cálculo — depende de `json` e `enum` e nada mais |
| `manifest.py` (182) | contrato de metadados de um run, sua decodificação do disco, e a identidade do conjunto de centros | `Path` → `PointRepresentationRunManifest`; mapa + centros → `sha256:…` | é o contrato que outras capabilities realmente leem; recusa schema não suportado | escrita, estatística, e qualquer import que impeça abrir um run com a stdlib |
| `writer.py` (504) | acumular resultados sob as invariantes do run, recusar publicar um stream que não bate com as métricas, publicar atomicamente | `EncodedRepresentation` \| `FailedSupport` → diretório publicado + manifest | índice de linha do payload, offsets JSONL e contagens do manifest são **uma** invariante | regra científica de outro dono, seleção de centros, encoding |
| `reader.py` (203) | abrir um run finalizado e responder **por identidade**, sem carregar o resto | `Path` + id → espaço, representação, vetor, falhas, integridade | `__DEBUG__/` é recusado explicitamente na leitura de registro | escrita, struct packing de saída, numpy, a geometria |
| `metrics.py` (71) | **como** as figuras do run são computadas | valores → distribuição, contagem por razão de falha, resumo por componente | o p95 é nearest-rank, uma definição científica com dono visível | decidir quando ou onde escrever; layout |

## `backends/` — encoders concretos

| Arquivo | Responsabilidade | Entrada → saída | Conceitos | Não deve possuir |
| --- | --- | --- | --- | --- |
| `__init__.py` (9) | declarar que este pacote **não exporta nada** | — | um backend é alcançado pelo próprio caminho de módulo, só do composition root | qualquer re-export |
| `geometric_descriptor.py` (295) | o descritor determinístico de 14 features: a linha de base científica | `PreparedSupport` → `EncodedVector` de 14 dims, `float64` | definição versionada (`DESCRIPTOR_VERSION`); regras explícitas de componente indefinido | numpy, torch, qualquer dependência de modelo |

### `backends/ptv3/` — o encoder aprendido opcional

Um módulo por motivo de falha. Importar o pacote carrega só `adapter.py`, que é stdlib puro.

| Arquivo | Responsabilidade | Entrada → saída | Conceitos | Não deve possuir |
| --- | --- | --- | --- | --- |
| `__init__.py` (32) | re-exportar **só** a superfície do adapter | `adapter` → 8 nomes com `__all__` | peso de import zero é o que mantém o backend opcional | o runtime, o backbone, a voxelização, os helpers de checkpoint |
| `adapter.py` (356) | a fronteira do backend: configuração, porta do runtime, erros, telemetria, encoder | `PreparedSupport` + `PTv3Runtime` → `EncodedVector` | é uma fronteira, não um método de treino; recusa famílias destiladas | torch, numpy, device, checkpoint loading, tensor |
| `pointcept_backbone.py` (151) | a arquitetura de um backbone PT-v3m1 publicado e a recusa de configuração que ele não honra | `PTv3Config` → `None` ou erro; → kwargs de construção | muda quando se adota outro checkpoint; as três divergências treino↔inferência vivem aqui | torch, numpy, Pointcept, device, pesos |
| `voxelization.py` (117) | mapear um suporte para a entrada esparsa do backbone e reduzir a saída a um vetor | coordenadas em metros → `Voxelization`; features por voxel → um vetor | dono único de `XYZ_CHANNELS`; parte do `input_definition` do espaço | torch, device, modelo — numpy fica dentro das funções |
| `checkpoint.py` (120) | identidade e confiança do checkpoint | `Path` + hash esperado → `None` ou erro; `state_dict` → pesos de backbone | sha256 fixado verificado **antes** de qualquer unpickle; head de classificação descartada de propósito | import de torch (recebe-o por parâmetro), arquitetura, inferência |
| `pointcept_module.py` (94) | o **único** módulo autorizado a importar biblioteca opcional ou a escrever em `sys.modules` | nome de módulo → módulo ou `PTv3RuntimeUnavailableError` | o side effect de processo está declarado na docstring, é idempotente, e existe para contornar os imports de treino do Pointcept | inferência, arquitetura, checkpoint, qualquer regra de domínio |
| `pointcept_runtime.py` (269) | manter o backbone residente num device, rodar um suporte, contabilizar memória, e traduzir cada falha de device no erro certo | coordenadas + `PTv3Config` → `PTv3Inference` | a ordem de `_ensure_loaded` é contrato: hash **antes** de qualquer import de torch | regra de domínio, voxelização, verificação de checkpoint, injeção de stub |

## Como usar este mapa em review

1. A função nova cabe na responsabilidade declarada do arquivo? Se não, qual arquivo a possui?
2. Se nenhum arquivo a possui, é mesmo desta capability, ou de `geometric_mapping`,
   `sensor_association`, `semantic_fusion` ou `runtime`?
3. Ela aparece na coluna "Não deve possuir" de algum arquivo? Então é violação, não preferência.
4. Ela adiciona um import proibido por [`architecture.md`](architecture.md)? Em particular: torch,
   numpy ou Pointcept fora de `backends/ptv3/`, ou `sensor_association` fora de
   `center_selection.py`.
5. Ela quebra uma das dez invariantes de [`testing.md`](testing.md)? Em especial: um run artifact
   ainda abre com a biblioteca padrão, e importar um backend ainda não carrega biblioteca de modelo?
6. Ela mistura mudança de comportamento dentro de reorganização? Nesse caso precisa de teste próprio
   e de explicação separada.
