# correction-benchmark-10_31-20260930 — benchmark B1, baseline E0, gate C10 e braços SAM2 (C4)

Bundle do **C0** do plano de correções (`~/.claude/plans/pasted-content-id-d1f2-quero-que-refactored-blossom.md`):
um benchmark reproduzível de um frame, o baseline **E0** medido por um extrator único, o
**gate C10** que compara qualquer braço com o E0 e os braços GPU de **regiões (C4)**. Tudo é
local e não commitado. Nenhum artefato de `workspace-old/` foi alterado; as saídas volumosas
dos braços ficam em `outputs/correction-benchmark-10_31-20260930/` (gitignored).

Convenção: **medido** = número produzido aqui; **hipótese** = mecanismo ainda não isolado.

## Donos das subpastas

| pasta | conteúdo | dono |
|---|---|---|
| `README.md`, `manifests/`, `annotations/`, `scripts/`, `results/` | benchmark, E0, extrator, gate, C4, variância do Qwen | este bundle (C0/C4/C10) |
| `calibration/`, `deskew/`, `regions/`, `association/` | C2, C3, métricas de região (TDD), C5 | outras frentes; não editadas aqui |

## B1 — o benchmark

Fonte: `manifests/benchmark.json` (gerado por `scripts/build_manifests.py`).

- **Frame:** 10:31 = `camera_1_image_raw-000055` (índice 55 na janela), header
  `1724621714.984856033` (segundo de gravação 630,985). Scan mais próximo
  `velodyne_points-000022`, header −50,26 ms antes da imagem. Pose GT por interpolação,
  bracket 302,6 ms.
- **Janela de ingestão:** 628–637 s (`--start-offset-s 631 --end-offset-s 634 --padding-s 3`;
  tempo de gravação 1724621712,66–1724621721,66), SequenceArtifact
  `aa0551d92f7b4a7aa3508dbaa1ee8ea3` (216 imagens, 89 scans, 1798 IMU) e pose
  `d422227f15f14d0cb8e071652c5c20b5`.
- **Mapa:** 81 scans aceitos (8 rejeitados pelo gap de pose > 400 ms). Braço **raw**
  (principal): 2.322.324 pontos. Braço **voxel** (secundário, `ScanVoxelPolicy` 0,10 m por scan):
  402.626 pontos.
- **Parâmetros congelados**, literais de `experiments/frame-10_31-raw-vs-voxel-20260930/run_two_arms.py`:
  ego-mask `examples/v0.2.0/corridor-02-image-masks.toml`; preset `canonical/1` sem
  `scene_interpretation`; prompt `region-label/v1`; view `tight_crop`; SAM2.1-hiera-tiny 0/0;
  Qwen3-VL-4B 4-bit T=0; cauda (mapping/resolução/relações) construída no driver com os valores
  canônicos do `e2e-real-canonical-run`.
- **Pilha de configuração** (ordem de precedência): `canonical-config.toml` →
  `canonical-2-empirical-validation-20260929/config/base.template.toml` → `providers.toml` →
  `corridor-02-image-masks.toml` → `region-label.toml`. Evidência de que a pilha atual é a do
  E0: o braço `E0-sam2` recompõe os fingerprints do E0 (SAM2 `9336bb27…`, normalização
  `c2a42c0d…`, passes `f1c08c08…`) e reproduz as 83 máscaras com o mesmo sha256.
- **Dataset:** sha256 dos YAMLs e do GT (o do GT confere com o que a ingestão registrou);
  bag de 24,5 GB por caminho + tamanho + o hash que a ingestão registrou (`hash_source=true`);
  zip de 15,6 GB por caminho + tamanho (nenhum manifesto registra o hash dele).
- **Identidade de código:** `git rev-parse HEAD` + sha256 de `git diff HEAD` + sha256 dos
  arquivos não rastreados que o run usa (`bench_common.code_identity`). O snapshot vai no
  manifesto e no `arm.json` dos reruns r3 e x1–x3; os braços C4 e os reruns r1/r2 rodaram
  minutos antes de o campo existir, com outras frentes editando `src/` em paralelo; o registro
  deles é o snapshot do manifesto mais os fingerprints de componente gravados nos artefatos
  (e a igualdade em bytes com o E0: máscaras do E0-sam2, respostas dos x). O E0 em si rodou numa árvore suja sobre
  `a3a175f09` cujo diff não foi registrado na época (limite conhecido).

**B2** (só C8/C9): `manifests/b2-proposal.json` é uma **proposta** (decisão D1 pendente): mesma
ingestão, imagens 37, 43, …, 79 (a cada 6, ~0,25 s), todas existem na sequência, estão dentro
dos limites de tempo do mapa e têm bracket de pose ≤ 302,6 ms.

## E0 — baseline

O E0 são os **runs existentes** `workspace-old/frame-10_31-raw` e `-voxel`, que são a
configuração baseline atual; não foram re-executados de ponta a ponta. Métricas em
`results/E0/metrics.json` e `results/E0-voxel/metrics.json`. Reprodução dos números da
investigação (`scripts/test_benchmark.py` confere cada um):

| métrica | investigação | E0 raw | E0 voxel |
|---|---|---|---|
| `geometry.bin` | `sha256:d4e4b03e…` / `b243d227…` | `sha256:d4e4b03e…` (= manifest) | `sha256:b243d227…` (= manifest) |
| pontos do mapa | 2.322.324 / 402.626 | 2.322.324 | 402.626 |
| regiões aceitas | 83 | 83 | 83 (mesma percepção) |
| claims / falhas de parse | 77 / 6 | 77 / 6 | 77 / 6 |
| pontos sem cobertura (esmaecidos no viewer) | 99,07 % / 98,97 % | 99,07 % | 98,97 % |
| associados em ≥ 2 máscaras | 92,3 % | 92,33 % (19.946/21.602) | 89,38 % |
| fragmentação / ≥ 80 % dentro de outra | 68,7 % / 43 | 68,7 % / 43 | idem |
| pixels cobertos por 1 máscara | 15 % | 15,1 % (25.455/168.168) | idem |
| labels com cor/material / entropia | 42/77 / 4,90 bits | 42/77 / 4,90 bits | idem |
| entidades no mapa / com label única | 38 / 27 | 38 / 27 | 33 / 23 |
| pontos únicos em entidades | 21.594 | 21.594 | 4.122 |

Nada diverge. O extrator também recontou a associação a partir de `geometry-support.u32`
(independente do diagnóstico do estágio) e bate: 21.602 associados, 19.946 em ≥ 2 observações.
A ocupação global de voxels de 0,10 m é 98.399 nos dois braços — esperado, porque o centroide de
uma célula por scan cai dentro da própria célula.

## Extrator de métricas (`scripts/extract_metrics.py`)

Um diretório de run → `metrics.json` (`correction-benchmark/metrics/v1`), por camada:

- **geometria:** sha256 do `geometry.bin` (e se bate com o inventário), `map_id`, contagem de
  pontos/scans, `calibration_identity`, contagem de voxels da agregação (só no voxel) e ocupação
  global a 0,10 m;
- **calibração:** resíduo de reprojeção nas correspondências anotadas → `pending_annotation`;
- **regiões:** `contextmap.evaluation.region_structure` (existe; criado por outra frente) com
  contenção 0,8 e duplicata IoU 0,8, + auditoria da descoberta (propostas do SAM2, rejeições
  por motivo, merges) + sha256 de cada máscara; se o módulo sumir, uma portabilidade de
  `p6_mask_quality.py` com os mesmos nomes de campo. Recall de objetos anotados →
  `pending_annotation`;
- **associação:** contagens por estágio, pares de pertencimento, fração multi-máscara, fração
  sem cobertura, recontagem independente;
- **semântica:** claims, falhas de parse, labels únicas, entropia, fração com prefixo de
  cor/material (léxico **só de avaliação** de `p7_label_provenance.py`), categorias, digests das
  views e dos prompts; `scene_label_rate` e `hallucination_rate` → `pending_annotation`;
- **entidades:** supports, entidades semânticas/resolvidas/no mapa, status, observações por
  entidade, taxa de isoladas (trivialmente 100 % em B1: um frame), labels por entidade;
- **relações:** contagens por estado (contexto).

Um braço parcial mede só as camadas que tem (as outras saem `not_in_run`).

O caminho com anotação já foi exercitado com uma anotação sintética de resposta conhecida
(`test_benchmark.py`): a correspondência do teste D da investigação (índice 481273, pixel base
(120,49; 76,33)) dá resíduo < 0,001 px sob a calibração do E0, tanto por `geometry_index` quanto
por `point_map_m` — o extrator e o pipeline usam a mesma convenção de centro de pixel; uma caixa
no bbox da region-0009 é recuperada por ela; "flooded hallway" conta como label de cena.

## Gate C10 (`scripts/gate.py`, `manifests/gate-policy.json`)

`gate.py --baseline results/E0/metrics.json --arm <braço>/metrics.json [--arm-kind …] [--target-layer …]`

- `arm_kind = none` (C4–C9): `geometry.bin`, `map_id`, contagem de pontos e identidade de
  calibração iguais ao E0;
- `calibration` (C2): `geometry.bin` byte-idêntico, `calibration_identity` e `map_id` novos;
- `geometry` (C3): bytes podem mudar; se mudarem, `map_id` novo;
- coerência interna (a associação usa o mapa e a calibração do braço) e checagens
  obrigatórias (sha256 = inventário; recontagem = diagnóstico);
- não-regressão das camadas **anteriores** à camada-alvo, com tolerância declarada (tolerância
  `null` = só reportar; as de semântica esperam a variância do Qwen).

Imprime os deltas por camada; saída 0/1/2. Testado em `scripts/test_benchmark.py` (15 testes, com a sanidade do E0 e um caminho de anotação sintética de resposta conhecida):
E0 vs E0 passa; `geometry.bin` adulterado falha; o voxel real falha como mudança não declarada
e passa como `geometry`; braço de calibração com bytes iguais e identidade nova passa, com
identidade velha ou bytes novos falha; regressão numa camada anterior falha e na camada-alvo
é só delta.

## C4 — braços SAM2 (`results/C4-sam2-grid/`)

Um processo por braço, executor real do runtime (`VisualPerceptionExecutor`), mesma pilha do E0
e mesma imagem preparada/ego-mask; a única variável é `predicted_iou_threshold`/
`stability_threshold` (override declarado no `arm.json`), preset reduzido a
`image_preparation` + `region_discovery`. Cada braço tem `metrics.json`, `gate.json`
(passa: nenhum braço toca a geometria), `masks_numbered.png` e `mask_overlap_count.png`.
Tabela completa: `table.md`/`table.csv`.

| braço | iou | estab. | propostas AMG | máscaras | fragmentação | dentro de outra | pixels em ≥ 2 máscaras | prof. máx. | cobertura da área útil |
|---|---|---|---|---|---|---|---|---|---|
| E0-sam2 | 0 | 0 | 431 | 83 | 0,687 | 43 | 142.713 (84,9 %) | 7 | 0,708 |
| E2 | 0,8 | 0,95 | 16 | 9 | 0,222 | 0 | 54 (0,2 %) | 2 | 0,092 |
| E3-g | 0,5 | 0,80 | 113 | 61 | 0,443 | 10 | 16.901 (12,9 %) | 4 | 0,554 |
| E3-g | 0,5 | 0,90 | 73 | 43 | 0,302 | 3 | 3.910 (5,6 %) | 2 | 0,295 |
| E3-g | 0,5 | 0,95 | 19 | 11 | 0,273 | 0 | 54 (0,2 %) | 2 | 0,102 |
| E3-g | 0,6 | 0,80 | 98 | 58 | 0,448 | 8 | 11.160 (8,8 %) | 3 | 0,534 |
| E3-g | 0,6 | 0,90 | 69 | 41 | 0,293 | 3 | 3.843 (5,6 %) | 2 | 0,289 |
| E3-g | 0,6 | 0,95 | 19 | 11 | 0,273 | 0 | 54 (0,2 %) | 2 | 0,102 |
| E3-g | 0,7 | 0,80 | 77 | 47 | 0,447 | 7 | 10.030 (8,4 %) | 3 | 0,503 |
| E3-g | 0,7 | 0,90 | 59 | 36 | 0,306 | 3 | 3.826 (5,8 %) | 2 | 0,279 |
| E3-g | 0,7 | 0,95 | 18 | 10 | 0,200 | 0 | 54 (0,2 %) | 2 | 0,093 |
| E3-g | 0,8 | 0,80 | 56 | 35 | 0,400 | 2 | 3.660 (4,5 %) | 3 | 0,339 |
| E3-g | 0,8 | 0,90 | 46 | 29 | 0,345 | 1 | 1.184 (2,4 %) | 2 | 0,204 |
| E3-g | 0,8 | 0,95 | 16 | 9 | 0,222 | 0 | 54 (0,2 %) | 2 | 0,092 |

Medido:

- **Determinismo:** `E0-sam2` reproduz as 83 máscaras do E0 com o mesmo sha256 e a mesma
  auditoria (113 merges, 173 rejeições de normalização, 175 de orçamento); `E2` e a célula
  0,80×0,95 da grade (dois processos) dão as mesmas máscaras (e o mesmo run id).
- **Os limiares encolhem a cobertura muito antes de zerar a sobreposição.** Com estabilidade
  0,95 sobram 9–11 máscaras e ~10 % da área útil, qualquer que seja o IoU predito; a
  estabilidade domina o efeito. Nas figuras, o E2 perde as duas portas azuis
  (`E2-iou0.80-stab0.95/masks_numbered.png`) — inspeção visual, não métrica.
- **16 ≠ 10:** a investigação contou 10/431 propostas que *passariam* 0,8/0,95 como filtro
  posterior do run 0/0. Com os limiares no gerador saem 16, porque o
  `SAM2AutomaticMaskGenerator` filtra por limiar **antes** do NMS (verificado no código:
  `_process_batch` filtra, `_process_crop` aplica `batched_nms`), então propostas que o NMS do
  0/0 suprimia sobrevivem.

**Seleção: `pending_annotation`.** O critério declarado antes de rodar é: (1) descartar braços
que perdem objeto anotado que o E1c/E0 encontra com IoU ≥ 0,5; (2) minimizar pixels em ≥ 2
máscaras; (3) desempate por fragmentação e depois menos máscaras. Sem a anotação o passo (1)
não é avaliável, e aplicar só o (2) escolheria trivialmente o braço mais restritivo — que é o
risco que o (1) existe para barrar. Nenhum braço foi escolhido.

Notas:

- As métricas 2D dos braços valem para o E1c também: calibração (C2) e deskew (C3) não mudam
  a imagem nem a descoberta. A métrica 3D (pontos em ≥ 2 máscaras) depende da associação e é
  do C5 (`association/`); basta rodar a associação com as regiões do braço e o mesmo extrator.
- **E3-L não rodou:** o cache HF offline tem `facebook/sam2.1-hiera-large` só no formato
  `transformers` (`model.safetensors`); o provider (`build_sam2_hf`) precisa de
  `sam2.1_hiera_large.pt`, ausente. Trocar para outra implementação mudaria mais de uma
  variável. Para rodar: baixar o `.pt` para o cache e
  `c4_sam2_arm.py --arm-id E3-L-… --checkpoint facebook/sam2.1-hiera-large --model-version 665f8e2a… --pred-iou … --stability …`
  com os limiares do braço selecionado.

## Variância do Qwen (#556) — `results/E0-qwen-variance/`

Viável sem mudar código de produção: `scripts/qwen_rerun.py` roda o executor real com a pilha
do E0 e o preset do E0 menos as features (DINOv2/CLIP não alimentam o Qwen:
`region_interpretation` depende só de `{image, regions}`). As regiões ficam congeladas porque
o SAM2 é determinístico (conferido a cada rerun: 83/83 máscaras com o sha256 do E0). Antes de
comparar saídas, `qwen_variance.py` confere a **entrada** do modelo por região: sha256 das views
(idênticos ao E0 em 83/83) e o prompt renderizado.

Achado no caminho (medido): o prompt renderizado embute o `perception_result_id`, que contém o
run id. Dois runs com run ids diferentes nunca mandam o mesmo prompt em bytes, mesmo com
configuração e região idênticas. Por isso há dois grupos:

- **r1–r3:** run id próprio por rerun (prompt = E0 a menos do run id);
- **x1–x3:** request idêntico ao do `run_two_arms.py` (`config_digest="perc"`,
  `content_hash="sha256:driven"`), mesmo run id do E0 → prompt e views byte-idênticos ao E0.

Medido (`summary.json`, `per_region.csv`; entrada idêntica ao E0 em 83/83 regiões em todos os
reruns, a menos do run id nos r):

| grupo | respostas brutas = E0 (por rerun) | desfecho igual em todos | label igual em todos | category | atributos |
|---|---|---|---|---|---|
| x1–x3 + E0 (prompt byte-idêntico) | 83, 83, 83 de 83 | 83/83 | 77/77 | 77/77 | 77/77 |
| r1–r3 (run id próprio) | 53, 53, 59 de 83 | 82/83 | 72/77 (93,5 %; par a par 95,7 %) | 63/77 (81,8 %) | 48/77 (62,3 %) |
| E0 + r1–r3 + x1–x3 (7 execuções) | — | 81/83 | 71/76 (93,4 %) | 62/76 | 42/76 |

Dispersão das métricas de semântica entre E0 e r1–r3: claims 77–78, falhas de parse 5–6,
labels únicas 37–39, prefixo cor/material 41–44 (0,532–0,571), entropia 4,83–4,95 bits,
categorias 14–16. Os x reproduzem o E0 exatamente (77, 6, 38, 42, 4,90, 14).

Leitura:

- **Com a mesma entrada em bytes, o Qwen é determinístico nesta máquina** (T=0, 4-bit, mesma
  pilha): três reruns reproduzem as 83 respostas brutas do E0 byte a byte.
- **A variação medida vem da entrada, não da decodificação:** mudar só o run id embutido no
  prompt (32 caracteres hex) altera 24–30 das 83 respostas brutas; contra o E0, cada r mantém
  72–73 das 76–77 labels primárias. As trocas são entre rótulos vizinhos
  (`door`↔`blue door`, `metal handle`↔`metallic handle`, `metal rail`→`metal handle`/`metal rod`,
  `door handle`→`metal handle`/`door`) e em regiões que oscilam entre claim e falha de parse
  (0023, 0074). Isso é medido para este frame e este prompt (`region-label/v1`).
- **Hipótese (não verificada):** os 32,2 % de concordância do #556 viriam, ao menos em parte, do
  mesmo mecanismo (dois runs com run ids diferentes → prompts diferentes), somado a um prompt de
  legendas longas (`region/v1`) mais sensível a perturbação. Os runs do #556 não foram
  examinados aqui.
- Consequência para as ablações do VLM (C6/C7): braços com configurações diferentes têm run ids
  diferentes, então o piso de ruído entre braços é o do grupo **r**, não o do x. As tolerâncias
  de semântica do gate (`manifests/gate-policy.json`) foram fixadas pela faixa E0 + r1–r3:
  falhas de parse ±1, fração com prefixo ±0,039.
- O E0 fica "dentro da variância medida" (aceite do C0): 77 claims no E0 contra 77–78 nos reruns.

Os `metrics.json` de cada rerun estão em `results/E0-qwen-variance/<rerun>/`.

## Regressão permanente "a semântica não altera a geometria"

`tests/end_to_end/test_semantics_do_not_change_geometry.py` (fora do bundle, na suíte), sobre o
harness sintético `tests/end_to_end/chain.py` (todos os estágios reais; só a percepção é
enlatada):

1. o `GeometricMapArtifact` construído só com ingestão → estado → mapa (semântica desligada) e o
   da cadeia completa têm `geometry.bin` e todos os `outputs/` byte-idênticos, mesmo `map_id`,
   `run_id`, contagem e inventário, e o mapa ainda verifica depois dos estágios semânticos;
2. os artefatos semânticos só referenciam o mapa: `geometry-reference.json = {map_id,
   point_count}`, toda referência de entidade decodifica para um índice < `point_count`, e o
   `geometry-support.u32` da associação também;
3. nenhum artefato semântico guarda XYZ próprio: os únicos campos com trincas de coordenadas
   são resumos derivados (`centroid_m`, `minimum_m`, `maximum_m`, `extent_m`, `up_direction`),
   nunca uma lista de pontos; os binários são só índices `.u32`; e os centroides/caixas das
   entidades e dos supports recomputam exatamente a partir dos pontos referenciados.

Os três passam (o invariante já vale; é uma promoção do teste E da investigação, sem mudança de
produção). Checagem de mutação feita fora da suíte: reescrever um arquivo do mapa depois da
cadeia, desalinhar o `point_count` da referência, pôr uma lista de pontos num support ou um
campo `anchor_xyz_m` numa entidade — cada teste falha no seu caso.

## Matriz de ablação (plano, seção 8) e como reproduzir

| braço | Cn | variável | estado aqui | como reproduzir |
|---|---|---|---|---|
| E0 | C0 | baseline | **medido** (runs existentes) | `extract_metrics.py --run-dir workspace-old/frame-10_31-raw --arm-id E0 …` |
| E0-qwen r/x | C0 | rerun do Qwen (N=3 ×2) | **medido** | `qwen_rerun.py --rerun-id r1` / `--rerun-id x1 --replicate-e0-identity`; `qwen_variance.py` |
| E1 | C2 | calibração refinada | pendente (C2, `calibration/`) | re-ingerir com o refinamento → `run_two_arms.py` → extrator → `gate.py --arm-kind calibration --target-layer calibration` |
| E1d | C3 | deskew | pendente (C3, `deskew/`) | mapa com deskew → cauda → extrator → `gate.py --arm-kind geometry --target-layer geometry` |
| E1c | C2+C3 | baseline combinado | pendente | as duas mudanças; gate `geometry` |
| E2 | C4 | SAM2 0,8/0,95 | **medido (2D)** | `c4_run_grid.sh` → `c4_evaluate.py` |
| E3-g{i} | C4 | grade 4×3 | **medido (2D)** | idem |
| E3 | C4 | limiares escolhidos | **pendente da anotação** | critério em `results/C4-sam2-grid/summary.json` |
| E3-L | P4 | SAM2.1-large | bloqueado (checkpoint `.pt` fora do cache) | `c4_sam2_arm.py --checkpoint …` |
| E3-G | P4 | máscaras GroundingDINO+SAM2-large do V1 | fora deste bundle | comparação offline só de métricas de região |
| E3-A{1..3} | C5 | candidatos/range/oclusão | C5 (`association/`) | `assoc.py` validado + extrator |
| E4, E5-B/C/D | C6 | view do VLM (N=3) | pendente; B/D exigem `VisualViewKind` novo | `qwen_rerun.py` com `view_policy` do braço |
| E5-G | P4 | Gemini Robotics-ER | pendente (chave, D8) | — |
| E5-P / E5-Pa | C7 | prompt estruturado / propagação | pendente | — |
| E6, E7-*, E8, E8-sep | C8/C9 | resolução multi-view, crença | exigem B2 (D1) | `manifests/b2-proposal.json` |

Todo braço novo: gravar `arm.json` com a variável, `arm_kind` e `target_layer`, rodar o
extrator e o gate contra `results/E0/metrics.json`.

## Reproduzir

```bash
PY=~/.cache/contextmap2-ablation/venv/bin/python      # GPU/modelos
PROV=~/.cache/contextmap2-ablation/providers
MPY=~/.local/share/mise/installs/python/3.12.14/bin/python3.12   # métricas
B=experiments/correction-benchmark-10_31-20260930
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# manifestos, imagens de anotação, E0
PYTHONPATH=src $MPY $B/scripts/build_manifests.py
PYTHONPATH=src $MPY $B/scripts/prepare_annotation_assets.py
PYTHONPATH=src $MPY $B/scripts/extract_metrics.py --run-dir workspace-old/frame-10_31-raw \
  --arm-id E0 --out $B/results/E0/metrics.json
PYTHONPATH=src $MPY $B/scripts/extract_metrics.py --run-dir workspace-old/frame-10_31-voxel \
  --arm-id E0-voxel --out $B/results/E0-voxel/metrics.json

# C4 (GPU, ~10 s por braço) e avaliação
bash $B/scripts/c4_run_grid.sh
PYTHONPATH=src $MPY $B/scripts/c4_evaluate.py

# variância do Qwen (GPU, ~5 min por rerun)
for r in r1 r2 r3; do PYTHONPATH=src:$PROV $PY $B/scripts/qwen_rerun.py --rerun-id $r; done
for r in x1 x2 x3; do PYTHONPATH=src:$PROV $PY $B/scripts/qwen_rerun.py --rerun-id $r --replicate-e0-identity; done
PYTHONPATH=src $MPY $B/scripts/qwen_variance.py

# gate e testes
$MPY $B/scripts/gate.py --baseline $B/results/E0/metrics.json --arm <braço>/metrics.json
PYTHONPATH=src $MPY -m pytest -q -p no:cacheprovider $B/scripts/test_benchmark.py \
  tests/end_to_end/test_semantics_do_not_change_geometry.py
```

## Pendências

- **Anotação fraca humana do frame 55** (`annotations/README.md`): desbloqueia a seleção do
  C4, o resíduo de reprojeção do C2 e as taxas de label de cena/alucinação do C6.
- Tolerâncias do gate: falhas de parse e prefixo cor/material já fixadas pela variância do
  Qwen; `scene_label_rate`/`hallucination_rate` seguem `null` até haver anotação.
- E1/E1d/E1c dependem do C2/C3; a métrica 3D dos braços C4 depende do C5.
- O E0 não foi re-executado de ponta a ponta nesta sessão: a percepção 2D foi reproduzida bit a
  bit (E0-sam2) e o Qwen, re-executado; geometria/associação/cauda são as do run existente.
- E3-L: baixar `sam2.1_hiera_large.pt` para o cache HF (não feito: sem instalação/download
  nesta sessão).

## Mudança fora do escopo sugerida (não feita)

- **Run id no prompt do VLM.** O prompt renderizado embute o `perception_result_id` (com o run
  id), então qualquer mudança de identidade do run muda a entrada do modelo e, medido aqui, 24–30
  de 83 respostas. Tirar os identificadores do texto do prompt (mantendo-os só nos metadados do
  request) tornaria reruns com configurações diferentes comparáveis sem esse ruído. O ponto é
  `src/contextmap/visual_perception/semantic_prompt.py:375` (`request_summary` renderizado no
  texto, com `perception_result_id`; `feature_id` também carregaria o run id se houvesse
  features no request). Muda o fingerprint dos prompts; precisa de issue, teste (Red: dois
  requests que diferem só no run id renderizam o mesmo texto) e decisão próprios.
