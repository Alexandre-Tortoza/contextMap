# Taxonomia de conflitos da associação 2D→3D

`contextmap.evaluation.association_conflicts` mede, para **um frame** e sem label, o
multi-pertencimento produzido por Sensor Association: um ponto visível do mapa entra em
**toda** máscara aceita que contém o seu pixel (`mask-membership-v1`). A pertença é evidência;
nada ali decide qual região é dona de um ponto. Este módulo mede de que tipo é a sobreposição e
se os pontos de uma máscara ficam numa superfície só, para que uma política futura mire a classe
dominante e evitável em vez de um palpite.

Ele não altera a pertença e não propõe posse. Uma hierarquia entre regiões é política
**a jusante** (formação de supports, C8), não deste módulo.

O módulo não está exportado em `contextmap.evaluation`; importe-o pelo submódulo. Depende só de
numpy e de `InlineMask`.

## Entradas

- `membership`: para cada região, os índices globais (únicos) dos pontos associados;
- `masks`: a máscara 2D de toda região da pertença, no mesmo espaço de imagem;
- `point_depth_m`: a profundidade de cada ponto em metros, indexada pelo índice global, numa
  só métrica (a do run: para MEI e fisheye, o alcance do raio, `RAY_RANGE`);
- `nested_containment_threshold`, `duplicate_iou_threshold`: obrigatórios, em `(0, 1]`;
- `depth_layers`: `DepthLayerCriterion`, também sem default.

## Definições (`ASSOCIATION_CONFLICT_METRIC_VERSION = "1.0.0"`)

**Multi-pertencimento.** Ponto multi-máscara = ponto associado a duas ou mais regiões.
`multi_mask_point_rate` = multi-máscara / pontos associados distintos; o report também dá a média
e o máximo de regiões por ponto e o histograma.

**Classe de um par.** Todo par não ordenado de regiões que compartilha ao menos um ponto é
classificado pelas máscaras 2D, nesta precedência:

1. `duplicate`: IoU `≥ duplicate_iou_threshold`;
2. `nested`: `|A ∩ B| / min(|A|, |B|) ≥ nested_containment_threshold` (a menor dentro da maior);
3. `partial`: o resto.

Duplicata vem antes de aninhada porque duas máscaras quase idênticas estão sempre aninhadas uma
na outra.

**Classe de um ponto.** Um ponto multi-máscara recebe a classe **mais severa** entre os pares a
que pertence: `partial` > `duplicate` > `nested`. Um ponto só de parte e todo (maçaneta dentro da
porta) conta como `nested`, não como conflito. As contagens por classe somam
`multi_mask_point_count`, então 100 % dos pontos multi-máscara ficam classificados; `partial` é a
classe residual.

O report dá, por classe: número de pares (`pair_count_by_class`), soma dos pontos compartilhados
pelos pares (`pair_shared_points_by_class`, um ponto conta uma vez por par) e pontos
multi-máscara pela classe mais severa (`multi_mask_points_by_class` e a taxa sobre os pontos
associados).

**Consistência de profundidade por máscara** (mesma definição de "contaminação por camadas" de
`p5_association_ablation.py`). As profundidades dos pontos da máscara, ordenadas, abrem uma nova
camada onde dois valores consecutivos distam mais que `max(gap_m, gap_ratio × mais próximo)`. A
camada principal é a **mais próxima** com ao menos `main_layer_min_share` dos pontos. A métrica é
a fração fora dela. Máscaras com menos de `min_points` pontos não são medidas (`None`).
`off_main_layer_fraction_weighted` é a média das medidas ponderada pelos pontos associados.
Extensão declarada em relação a p5: quando nenhuma camada atinge a fração mínima (p5 levantaria
`StopIteration`), a principal é a maior (a mais próxima no empate). Os parâmetros de p5 são
`gap_m=0,3`, `gap_ratio=0,08`, `main_layer_min_share=0,2`, `min_points=30`.

**Oclusão próxima** (`measure_occlusion_proximity`, definição de `p5_occlusion_anatomy.py`).
Para os pontos ocluídos, `gap = profundidade − profundidade de suporte` (mesma métrica). A métrica
é a fração com `gap ≤ proximity_m`. Mede quanto da oclusão pode ser a mesma superfície em ângulo
rasante ou engrossada pelo acúmulo; não decide isso. Todo ponto ocluído precisa de profundidade e
suporte finitos.

Taxas sem população são `None`, nunca `0.0`.

## Limitações

- As classes são 2D: dizem como as máscaras se sobrepõem, não se os pontos compartilhados estão
  na mesma superfície. A camada de profundidade é medida à parte, por máscara.
- A camada principal é uma heurística sem rótulo. Uma máscara cujo fundo tem mais de 80 % dos
  pontos faz do fundo a camada principal.
- A taxonomia do E0 é sensível ao limiar de aninhamento (abaixo). O limiar do plano (0,9) é
  estrito, e muitos pares têm contenção entre 0,5 e 0,9.

## Validação no E0 (frame 10:31)

`experiments/correction-benchmark-10_31-20260930/association/validate_association_conflicts.py`
lê a pertença persistida de `workspace-old/frame-10_31-raw` (API pública) e a profundidade por
ponto da reimplementação validada (`assoc.py`). Reproduz p5 exatamente:

- 21.602 pontos associados e 53.977 pertenças;
- 92,3 % multi-máscara, até 7 regiões por ponto;
- 39 máscaras com ≥ 30 pontos, contaminação ponderada 0,42 %, iguais por máscara;
- 12,19 % dos 1.516.151 ocluídos a ≤ 0,5 m do suporte.

Primeira taxonomia do E0, com aninhado ≥ 0,9 e duplicata ≥ 0,8:

| Classe | Pontos multi-máscara (taxa) | Pares |
|---|---|---|
| partial | 57,8 % | 169 |
| nested | 34,5 % | 35 |
| duplicate | 0 % | 0 |

A normalização já funde IoU ≥ 0,8, por isso não há duplicatas. Com aninhado ≥ 0,8, partial cai
para 55,3 %; com ≥ 0,5, cai para 19,8 %.
