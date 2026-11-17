# Diagnóstico offline do grafo de relações — canonical/2

Run analisado: `outputs/canonical-2-real-20260929/`
Script: `scripts/diagnose_relation_graph.py` (read-only, sem alterar produção nem os artefatos do run)
Dados brutos: `report.json`

Diagnóstico de 30 minutos sobre o run existente para separar as hipóteses
concorrentes com números, antes de tocar em calibração de aparência (H5, fora de
escopo aqui). Todas as verificações de sanidade passaram (ver `sanity_checks` no
`report.json`).

## Hipótese primária (reformulada)

> A baixa consolidação de entidades infla o número de nós e, por consequência, o
> número de pares geometricamente candidatos. A contribuição das superfícies
> estruturais e da política de candidatos foi **medida** aqui. **Não há base**
> para corrigir o problema com filtragem semântica dentro de `spatial_relations`.

O diagnóstico sustenta essa reformulação e corrige três afirmações que o brief e
a crítica traziam sem medição.

## Correções factuais (o que os números mostram)

### 1. A "mediana de grau 52" não se reproduz — o grau canônico é 18

O total de 98.568 relações é exatamente `2 × 49.284`: cada relação canônica
(`derived_from = None`) tem uma gêmea simétrica/inversa. Só `next_to` e
`touching` foram avaliados, ambos simétricos. Sobre o conjunto canônico:

| medida de grau (supported)                 | contagem | média | mediana | p95  | máx |
|---------------------------------------------|---------:|------:|--------:|-----:|----:|
| grau canônico, todos os 535 nós             |      535 | 18,12 |    18,0 | 45,0 |  83 |
| grau canônico, só nós não isolados (374)    |      374 | 25,93 |    25,0 | 50,4 |  83 |
| incidência sobre TODOS os registros (twins) |      535 | 37,10 |    36,0 | 92,6 | 168 |

Nenhuma medida canônica chega a 52. O valor do brief é compatível apenas com uma
contagem de incidências sobre registros duplicados (simetria dobra o grau
conceitual). O grau real por nó no grafo de suportadas é **mediana 18** (todos os
nós) ou **25** (não isolados). 161 dos 535 nós não têm nenhuma aresta suportada;
o maior componente conexo do grafo de suportadas tem 374 nós.

### 2. A vizinhança avalia 17,3% dos pares, não 34,5%

`candidates = 49.284` conta pares **(par, predicado)**, não pares de entidades.
Como `next_to` e `touching` podem ser candidatos para o mesmo par, os pares não
ordenados únicos avaliados são `49.284 / 2 = 24.642`, ou seja **17,3% de
`C(535,2) = 142.845`** — metade do valor citado. Pares suportados únicos: 4.848
(vs. 4.962 relações canônicas suportadas: 114 pares têm `next_to` **e**
`touching` suportados simultaneamente).

`by_state` canônico: supported 4.962 · rejected 7.304 · unresolved 37.018.
Por predicado: `next_to` (sup 4.099 / rej 1.795 / unr 18.748),
`touching` (sup 863 / rej 5.509 / unr 18.270).

### 3. Remover superfícies estruturais reduz a **contagem**, não a **densidade**

Classificação estrutural por palavra-chave sobre rótulos de texto livre
(sensibilidade, não verdade de campo — ver ressalva). Grafo de suportadas
(4.848 arestas, 535 nós, densidade 0,0339, mediana de grau 18):

| variante | estruturais | arestas → residual | queda de arestas | densidade base → residual | mediana grau base → residual |
|----------|------------:|--------------------|-----------------:|---------------------------|------------------------------|
| strict (`wall,floor,ceiling,pillar,column`)          | 225 (42,1%) | 4.848 → 2.347 | 51,6% | 0,0339 → **0,0490** | 18 → 16 |
| loose (+`structural,surface,beam,slab,girder`)       | 388 (72,5%) | 4.848 → 437   | 91,0% | 0,0339 → **0,0407** | 18 → 6  |

Decomposição strict das arestas suportadas: estrutural–estrutural 525,
estrutural–outro 1.976, outro–outro 2.347.

**Leitura:** a poda estrutural derruba a contagem absoluta de arestas (efeito
combinatório de remover metade dos nós), mas o subgrafo objeto–objeto **não fica
esparso**: a densidade *sobe* (0,0339 → 0,0490) e a mediana de grau quase não cai
(18 → 16). Ou seja, superfícies estruturais **não são o motor da densidade/grau**
por nó. Isso confirma, agora com medida, que um filtro semântico em
`spatial_relations` não é a correção arquitetural adequada para densidade — ao
mesmo tempo em que atende à exigência da crítica de medir grau residual, e não só
a queda percentual de arestas.

## H1 — fragmentação de entidades (causa dominante dos nós)

- 606 entidades de origem → 535 resolvidas.
- **484 singletons (90,5%)**; histograma de tamanho de grupo: {1:484, 2:34, 3:15,
  4:1, 5:1}; média de fontes por resolvida 1,13; máx 5.

A consolidação quase não ocorre porque só o canal de geometria foi avaliado na
resolução de entidades; aparência, semântica, temporal e point_representation
ficaram todos em `not_evaluated` para os 183.315 = `C(606,2)` pares (não é a
aparência isoladamente desligada, como o brief sugeria — é todo canal não
geométrico). Alta contagem de nós é o principal amplificador dos pares candidatos.

## H4 — materialização do grafo

O context map materializa **todas** as 98.568 relações como arestas:
supported 9.924 · unresolved 74.036 · rejected 14.608 → **89,9% não são
suportadas**, e mesmo o "supported" é a figura dobrada (9.924 = 2 × 4.962; apenas
4.848 pares únicos). Preservar avaliações negativas/indeterminadas é coerente com
evidence/belief/knowledge, mas o grafo de **conhecimento** poderia materializar
apenas as suportadas canônicas (4.848 pares), mantendo negativas, indeterminadas
e gêmeas derivadas em evidência/proveniência. Observação registrada; **nenhuma
mudança de código foi proposta neste experimento**.

## Ordem de investigação sustentada pelos dados

- **H1 fragmentação** — confirmada (90,5% singletons). Dominante para o número de nós.
- **H2 densidade de candidatos** — 17,3% de todos os pares avaliados; grau
  mediano 18 no grafo de suportadas. A vizinhança é ampla para um corredor;
  validar o `candidate gate` independentemente da resolução de entidades.
- **H3 multiplicador estrutural** — grande em contagem absoluta, **não** em
  densidade/grau. Não justifica filtro semântico em `spatial_relations`.
- **H4 materialização** — 89,9% do artefato não são relações afirmadas; candidato
  a materializar só suportadas no grafo de conhecimento.
- **H5 associação por aparência** — **próximo passo**, fora do escopo deste
  diagnóstico: construir conjunto de calibração e varrer o threshold por
  precision/false-merge dentro do gate geométrico.

## Ressalvas

- **H3 é sensibilidade, não verdade de campo.** Não existe tipo "superfície
  estrutural" no domínio; os rótulos são frases descritivas abertas do modelo. As
  flags vêm de casamento de palavra inteira sobre esses rótulos. A classificação
  por entidade está em `entity_structural_classification` no `report.json` para
  inspeção; as variantes strict/loose delimitam a sensibilidade.
- Números específicos deste run; para outro workspace, as asserções de sanidade
  viram apenas informativas.

## Reproduzir

```bash
PYTHONPATH=src python3 \
  experiments/relation-graph-diagnostic-20260929/scripts/diagnose_relation_graph.py \
  outputs/canonical-2-real-20260929 \
  experiments/relation-graph-diagnostic-20260929/report.json
```
