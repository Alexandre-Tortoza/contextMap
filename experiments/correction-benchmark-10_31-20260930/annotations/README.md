# Anotação fraca do frame 55 (piloto, B1)

Anotação humana de **piloto com rótulo fraco**: o número que ela produz é go/no-go, não aceite
(decisão do usuário no plano). Ela desbloqueia as métricas marcadas `pending_annotation` no
`metrics.json` de cada braço:

| métrica | camada | quem usa |
|---|---|---|
| `known_object_recall` (melhor IoU ≥ 0,5 por instância) | regiões | seleção do braço C4 (passo 1 do critério) |
| `reprojection_residual_px` (mediana/p90) | calibração | aceite (c) do C2 |
| `scene_label_rate`, `hallucination_rate` | semântica | C6/C7b |

## Arquivos

- `frame-55.png` — a imagem do frame, 640×480. É **byte-idêntica à imagem preparada** que o
  pipeline usou: sha256 `c00c9d53…` = `source_image_sha256` gravado nas views semânticas do E0
  (a preparação não redimensiona nem recorta).
- `frame-55-grid.png` — a mesma imagem com grade de 50 px e a área inutilizável escurecida
  (fora da vinheta válida e sobre o rig, pela ego-mask). Não anote nada na área escura: as
  regiões ali são rejeitadas antes de qualquer métrica.
- `frame-55.template.json` — o formato. Copie para **`frame-55.json`**, preencha e troque
  `"status": "template"` por `"status": "annotated"`. O extrator só usa o arquivo com
  `status = annotated`; sem ele, tudo que depende de anotação sai `pending_annotation`.
- Apoio 3D (fora do bundle, gitignored): `scripts/export_map_ply.py` gera
  `outputs/correction-benchmark-10_31-20260930/annotation/e0-raw-map-near-frame55.ply`, o mapa
  raw do E0 com o índice global de cada ponto como campo escalar `index`.

## Convenções

- Pixel: `u` = coluna, `v` = linha, 0-based, origem no canto superior esquerdo. Um inteiro
  `(u, v)` é o **centro** do pixel — a mesma convenção da projeção do pipeline
  (`pixel = floor(uv + 0,5)`). Leia a coordenada que o editor de imagem mostra para o pixel.
- Caixa: `box_xyxy = [x_min, y_min, x_max, y_max]`, semiaberta (`x_max`/`y_max` exclusivos),
  como `BoundingBox` do contextmap. Polígono: `polygon_xy = [[u, v], …]`, em ordem. Um dos
  dois basta; com os dois, o polígono vira a máscara.
- Cobertura **parcial**: o que não for anotado não é verdade negativa. Anote o que der no
  tempo do piloto, priorizando objetos inteiros e visíveis.

## O que anotar

1. **Instâncias** (`instances`), com `class` ∈ {`door`, `door_handle`, `window`, `wall`,
   `ceiling`, `floor`} e um `instance_id` único (`door-01`, `window-02`, …):
   - uma instância por objeto físico (duas portas = duas instâncias, mesmo se iguais);
   - parede/teto/piso: uma instância por superfície contínua visível (p.ex. `wall-left`,
     `wall-right`), sem tentar separar placas;
   - `accepted_labels`: rótulos livres que você aceitaria como corretos para aquela
     instância (p.ex. `["door", "blue door", "metal door"]`). O casamento é
     casefold-exato, sem sinonímia implícita: o que não estiver listado (nem a classe, com `_`
     trocado por espaço) conta como não-casamento;
   - `truncated`/`occluded` quando a instância sai da imagem ou está parcialmente coberta.
2. **Termos de cena** (`scene_terms`): rótulos que descrevem a cena e não um objeto
   (`hallway`, `corridor`, `flooded hallway`, …). Uma região pequena (< 50 % da imagem)
   rotulada com um termo de cena conta como label de cena.
3. **~10 correspondências 2D↔3D** (`correspondences`) em pontos bem definidos e espalhados
   (quinas de porta, janela, batente; evite bordas de oclusão e o chão molhado):
   - `pixel_uv` na imagem;
   - o ponto 3D escolhido **pela geometria**, no PLY acima: `geometry_index` (o `index` do
     ponto; `map_id` já vem preenchido com o mapa raw do E0) **ou** `point_map_m` `[x, y, z]`
     em metros no frame `map`;
   - **nunca** escolha o ponto 3D projetando-o com o pipeline: a correspondência mede a
     calibração e não pode depender dela;
   - `pixel_tolerance_px`: sua incerteza no pixel (2 px por padrão).

## Relação com `contextmap.evaluation.annotations`

O formato do repositório (`src/contextmap/evaluation/annotations.py`) cobre os mesmos níveis,
mas em documentos separados por família e com máscaras `InlineMask` (uma lista booleana por
pixel, 307.200 valores por instância a 640×480). Para um piloto manual, um único JSON com
caixas/polígonos é mais prático; os campos foram nomeados para converter 1:1 quando valer a
pena (sem conversor agora, YAGNI):

| aqui | família do repositório |
|---|---|
| `instances[].instance_id` + `box_xyxy`/`polygon_xy` | `regions/v1`: `RegionAnnotation(region_id, box, mask)`, `Coverage.PARTIAL` |
| `instances[].class` + `accepted_labels` | `semantics/v1`: `SemanticAnnotationRecord(concepts, status=LABELED)`, `casefold-exact/1` |
| `scene_terms` | `scene_context/v1`: `SceneContextAnnotation(attributes)` |
| `correspondences[]` | `geometry/v1`: `GeometryCorrespondence(pixel_uv, point_frame_id="map", point_m, pixel_tolerance_px)` — somar 0,5 a `u`/`v` ao converter, porque o repositório usa a grade semiaberta (centro do pixel em `i + 0,5`) |

A métrica de recall já usa o tipo do repositório: o extrator converte as instâncias em
`RegionAnnotation` e chama `contextmap.evaluation.region_structure.known_object_recall`.
