# Escopo — calibração da associação por aparência (H5)

Especificação (sem implementação) do experimento que decide se e como ligar o
canal `APPEARANCE` na resolução de entidades. Segue-se o diagnóstico
`experiments/relation-graph-diagnostic-20260929/`, que mostrou que a alavanca
real contra o tamanho do grafo é **reduzir nós** (fundir fragmentos), não filtrar
relações: 484/535 (90,5%) singletons porque a resolução rodou **só no canal de
geometria**.

Registrado antes de codar conforme AGENTS.md §22 (problema, justificativa,
entrada, processamento, saída, métrica, teste) e §9 (exceção de pesquisa:
hipótese, baseline, variável alterada, dataset, métrica, critério de aceite).

## 1. Problema

A resolução decide se duas entidades de origem são o **mesmo objeto físico**. Hoje
só a geometria vota; fragmentos do mesmo objeto vistos de ângulos/frames
diferentes não fundem. O canal de aparência existe mas está **desligado**
(commit `6788b83`) porque exige limiares de similaridade que dependem do espaço de
embedding e do domínio, e ainda não foram medidos. Uma similaridade de cosseno
não tem significado absoluto (AGENTS.md §5/§6) — o corte precisa de dados.

## 2. Hipótese (H5)

> Ligar o canal de aparência, com uma banda de similaridade calibrada dentro do
> gate geométrico, funde fragmentos do mesmo objeto e reduz nós/duplicatas
> **sem** elevar a taxa de falsos merges acima de um limite aceitável — e isso se
> propaga a jusante como menos pares candidatos e menos relações.

## 3. Baseline e variável alterada

- **Baseline:** o run atual `canonical-2-real-20260929`, geometry-only
  (`entity.false_merge.rate`, `entity.duplicate.rate` e nº de resolvidas medidos
  pelo `IdentityEvaluation`; nós/relações pelo diagnóstico de grafo).
- **Variável alterada (uma):** o profile passa a selecionar
  `entity_resolution.appearance` com uma `AppearanceComparisonPolicy` calibrada.
  Nada mais muda (mesma perception, mesmo candidate gate, mesmo `spatial_relations`).

## 4. Infra que já existe (não reimplementar — DIP/YAGNI)

- `AppearanceComparator` / `AppearanceComparisonPolicy`
  (`src/contextmap/entity_resolution/appearance_comparison.py`). A política é uma
  **banda de dois cortes**, não um limiar único:
  - `min_supporting_similarity` (≥ → aparência **apoia** o merge);
  - `max_conflicting_similarity` (≤ → aparência **contraria**; `None` = nunca
    contraria, pois um ângulo diferente pode parecer diferente);
  - entre os dois → ambíguo (sem voto). Agregação
    `physical-observation-prototype-v1`: um voto por observação física.
- `FeatureStoreVectorSource` carrega os vetores dos feature stores da perception.
- `IdentityAnnotationSet` + `IdentityEvaluation`
  (`src/contextmap/evaluation/`): verdade de campo por **ocorrência**
  `(sample, observation, region)`, com `PhysicalIdentity` e
  `DistinctIdentityPair`; métricas `entity.false_merge.rate`,
  `entity.duplicate.rate`, `pairwise_precision`, `pairwise_recall`,
  `same/distinct_pair_outcomes`. Identidade nunca é inferida de rótulo igual.
- Wiring de composição: `entity_resolution.appearance` é `_construct_optional`
  (`composition.py:1900`); ligar = o profile declarar o componente com config.

## 5. Espaço de embedding (medido neste run)

Duas coberturas region-scoped, ambas em **606/606** entidades (verificado):

| embedding_space_id (prefixo) | escopo             | provável origem            |
|------------------------------|--------------------|----------------------------|
| `sha256:1b1d054c…`           | dense (699) + region (1040) | DINOv2 denso + pooling p/ região |
| `sha256:70a72c65…`           | region (1040)      | CLIP region                |

O canal só compara features **region-scoped** (dense descreve a imagem toda). Há,
portanto, dois espaços candidatos. **Default do plano:** calibrar os dois e
comparar (custo baixo, ambos cobrem tudo), declarar o vencedor no
`embedding_space_id` da política. Correção a uma afirmação anterior: não é "DINO"
por padrão — é o espaço region-scoped que a calibração escolher.

## 6. Conjunto de calibração (dataset/seleção) — abordagem: **PILOTO com rótulo fraco**

Não existe `IdentityAnnotationSet` do corridor-02 no disco (nenhum arquivo de
identidade encontrado no repo). Decisão tomada: **piloto de primeiro sinal**, sem
verdade de campo completa, para verificar se a banda de aparência separa positivos
de negativos antes de investir em anotação exaustiva. O número do piloto **não é**
o número final de aceite (§9).

Duas partes, com custos bem diferentes:

- **Scoring (sem rótulo, roda já):** computar a distribuição do cosseno de
  aparência sobre **todos** os pares no gate geométrico, nos dois espaços
  region-scoped (§5). Só isso já diz se há estrutura (ex.: distribuição bimodal =
  há sinal separável; unimodal = aparência pouco discrimina neste domínio).
- **Rótulo fraco (mínimo trabalho humano):** anotar à mão um subconjunto pequeno
  de identidades **óbvias** — alguns pares claramente do mesmo objeto (positivos)
  e alguns claramente distintos (negativos), a partir de `labeled-png/`, `masks-png/`
  e `frames-png/` do run. Sobrepõe-se esse rótulo à distribuição de scores para ver
  onde caem positivos vs negativos.

Ressalvas do piloto (registrar no relatório):

- Amostra pequena e enviesada para casos fáceis → estima **separabilidade**, não a
  banda de produção nem a taxa real de falsos merges.
- Ainda vale a regra do harness (§ split): se o piloto evoluir para calibração de
  produção, o conjunto que escolhe a banda deve ser disjunto do que reporta o
  aceite. No piloto isso é anotado como limitação, não como resultado final.

## 7. Processamento

1. Enumerar pares **dentro do gate geométrico** (candidatos já plausíveis; não
   comparar aparência de coisas distantes).
2. Para cada par, calcular o cosseno via `AppearanceComparator` no
   `embedding_space_id` declarado (reusar `FeatureStoreVectorSource`).
3. Rotular cada par pela `IdentityAnnotationSet`: **mesma** identidade
   (positivo) vs **declarada-distinta** (negativo). Pares sem cobertura de
   anotação são excluídos, não contados.

## 8. Varredura da banda e critério de escolha

- Varrer uma grade de `(min_supporting_similarity, max_conflicting_similarity)`.
- Para cada banda: `pairwise_precision`, `pairwise_recall`, `false_merge_rate`.
- **Escolher** a banda que **minimiza a taxa de falsos merges** (erro caro e de
  difícil reversão — funde objetos distintos e corrompe o mapa) sob um recall
  mínimo aceitável. Escolha feita no set de calibração; números reportados no set
  held-out.

## 9. Métrica de sucesso / rejeição

Via `IdentityEvaluation` no conjunto held-out, ablação **geometry-only** vs
**geometry+appearance**:

- **Aceite:** cai o nº de resolvidas / `duplicate.rate` (menos fragmentação) com
  `false_merge.rate` dentro do limite (idealmente ~0); e o efeito a jusante
  aparece ao **rerodar `experiments/relation-graph-diagnostic-20260929`** (menos
  nós → menos pares candidatos → menos relações).
- **Rejeição:** se nenhuma banda reduz duplicatas sem elevar falsos merges acima
  do limite, o canal de aparência não é a alavanca — registra-se o resultado
  negativo e reabre-se H2 (candidate gate) / perception (super-segmentação).

## 10. Saídas e proveniência

- `AppearanceComparisonPolicy` congelada (`embedding_space_id`, os dois cortes,
  `fingerprint()`), pronta para o profile.
- Relatório com: id do conjunto de anotação, split, seed, commit, espaço(s)
  comparado(s), curva precision/recall/false-merge por banda, e a ablação de
  identidade + o delta do grafo.

## 11. Fora de escopo

Mudar perception/segmentação; alterar o candidate gate geométrico; qualquer
mudança em `spatial_relations`; combinar scores de canais (proibido — cada canal
vota separado).

## 12. Riscos / ressalvas

- Corredor tem superfícies repetidas (paredes/piso) → falsos "parecidos"; a banda
  precisa ser conservadora, daí priorizar false-merge rate.
- Os dois espaços podem discordar; por isso a comparação lado a lado.
- Sem a anotação (§6) o experimento não produz número defensável — é o próximo
  passo a destravar.

## Passos (ordem de execução) — modo PILOTO

1. **Scoring sem rótulo (roda já):** script que enumera pares no gate geométrico,
   calcula o cosseno via `AppearanceComparator` nos dois espaços (§5) e emite a
   distribuição de scores (histograma, quantis) por espaço. Read-only, no padrão
   de `experiments/relation-graph-diagnostic-20260929/`.
2. **Rótulo fraco (humano, mínimo):** pequeno conjunto de pares óbvios
   positivos/negativos a partir de `labeled-png/`/`masks-png/`.
3. **Sobreposição:** onde caem positivos vs negativos na distribuição → há banda
   separável? Decisão go/no-go para investir em anotação completa.
4. **Se go:** evoluir para `IdentityAnnotationSet` real + split, varredura da
   banda, congelar `AppearanceComparisonPolicy`, ablação de identidade no held-out.
5. **Efeito a jusante:** rerodar o diagnóstico de grafo; comparar nós/relações vs
   baseline.
