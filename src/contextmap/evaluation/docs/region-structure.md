# Estrutura das regiões de um frame

`contextmap.evaluation.region_structure` mede, sem anotação, a estrutura do conjunto de máscaras
de **um frame**: quantas são, de que tamanho, quão fragmentadas, quanto se empilham e quanto se
repetem. O objetivo é comparar configurações de Region Discovery (por exemplo, limiares de
qualidade do SAM2) pela estrutura antes que exista qualquer label. Opcionalmente, com objetos
anotados, mede se algum objeto conhecido deixou de ser encontrado.

O módulo não está exportado em `contextmap.evaluation` (a API pública do pacote não mudou);
importe-o pelo submódulo. Depende só de numpy e dos tipos públicos `InlineMask`,
`BoundingBox` (`visual_perception`) e `RegionAnnotation` (`evaluation.annotations`).

## Entradas

- `masks`: `Mapping[str, InlineMask]`, as máscaras do frame por `region_id`, todas no mesmo
  espaço de imagem. O chamador escolhe a população (tipicamente as regiões aceitas; num run
  persistido, `PerceptionRunReader.mask_store().load(...)`).
- `containment_threshold` e `duplicate_iou_threshold`: obrigatórios, em `(0, 1]`, **sem
  default**. O valor depende da pergunta, e o report o registra.
- `proposal_count` (opcional): propostas do backend de onde as máscaras saíram (no run, a soma de
  `BackendDiagnostics.proposal_count` da auditoria de discovery).
- `usable_area` (opcional): pixels onde uma região faz sentido (região válida menos exclusões).

## Definições (`REGION_STRUCTURE_METRIC_VERSION = "1.0.0"`)

| Métrica | Definição |
|---|---|
| `mask_count` | máscaras medidas |
| `accepted_mask_ratio` | `mask_count / proposal_count`; `None` sem `proposal_count` |
| `area_px_p10/median/p90` | quantis das áreas, método `linear` do numpy |
| componente | pixels ligados por vizinhos de **aresta** (conectividade 4; contato diagonal não une) |
| `fragmented_mask_count`, `fragmentation_rate` | máscaras com mais de um componente; sobre `mask_count` |
| fração de A dentro de B | `|A ∩ B| / |A|` |
| `masks_contained_in_another` | máscaras cuja maior fração dentro de outra atinge `containment_threshold` |
| `contained_pair_count` | pares **não ordenados** com `|A ∩ B| / min(|A|, |B|) ≥ containment_threshold` |
| `duplicate_pair_count` | pares não ordenados com IoU `≥ duplicate_iou_threshold` |
| `overlapping_pair_count` | pares não ordenados com ao menos um pixel em comum |
| profundidade de aninhamento | quantas máscaras cobrem o pixel |
| `nesting_depth_histogram`, `max_nesting_depth` | pixels por profundidade (≥ 1) e o máximo |
| `pixels_with_multiple_masks`, `multiple_mask_pixel_fraction` | pixels cobertos com profundidade ≥ 2; sobre os cobertos |
| `usable_area_coverage` | fração da área útil coberta por ao menos uma máscara |

Contido e duplicado são contagens independentes: um par duplicado normalmente também é contido.
Uma taxa sem população (nenhuma máscara, nenhum pixel coberto) é `None`, nunca `0.0`.

Por máscara (`MaskStructure`): área, componentes, maior fração dentro de outra e qual máscara a
atinge (`None` quando a máscara não toca nenhuma outra; em empate, a primeira em ordem de id).

A conectividade 4 é o elemento estruturante default do `scipy.ndimage.label` em 2D, usado pela
investigação (`p6_mask_quality.py`). A implementação usa só numpy: runs horizontais por linha,
arestas entre runs de linhas vizinhas cujos intervalos se sobrepõem, e propagação de rótulo
mínimo com atalho até o ponto fixo.

## Recall de objetos conhecidos

`known_object_recall(masks, objects, match_iou_threshold=...)` recebe `RegionAnnotation`s com id
único. Para cada objeto:

- anotado por **máscara**: IoU de pixel com cada máscara;
- anotado só por **caixa**: IoU entre a caixa anotada e a caixa justa (semiaberta) de cada
  máscara, porque a caixa não diz quais pixels pertencem ao objeto.

Com máscara e caixa, a máscara vale. O report dá o melhor IoU, a máscara que o atinge (`None` se
nenhuma sobrepõe), se o objeto foi recuperado (`best_iou ≥ match_iou_threshold`) e o recall.
Sem objetos, `recall` é `None`.

## Validação no E0 (frame 10:31)

`experiments/correction-benchmark-10_31-20260930/regions/validate_region_structure.py` lê as 83
regiões aceitas de `workspace-old/frame-10_31-raw` pelas APIs públicas e reproduz
`p6_mask_quality.json`: 68,7 % fragmentadas, 43/83 com ≥ 80 % dentro de outra, 15,1 % dos pixels
cobertos com uma só máscara, histograma de aninhamento idêntico (máximo 7), quantis de área,
13 pares com IoU ≥ 0,5 e cobertura da área útil de 70,8 %. A contagem de componentes coincide
com o `scipy.ndimage.label` nas 83 máscaras e em 400 máscaras aleatórias. Duas diferenças são
de definição, e estão explicadas no resultado:

- p6 conta pares contidos **ordenados** (57 = 56 não ordenados + 1 par com as duas direções
  ≥ 0,8); o módulo conta não ordenados (56);
- para uma máscara sem sobreposição, p6 grava o argmax de um vetor de zeros como "contida em";
  o módulo grava `None`.

## O que não mede

Qualidade semântica, correção de uma label, ou recall sem anotação. As métricas descrevem a
estrutura do conjunto; qual estrutura é "boa" depende do critério de seleção declarado pelo
experimento (C4), não deste módulo.
