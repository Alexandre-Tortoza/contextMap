# Validação da associação máscara RGB → pontos LiDAR (Fase 1)

Diagnóstico da associação 2D→3D (`sensor_association`) do ContextMap2, motivado por dois
sintomas relatados: (1) "não está pintando corretamente os pontos"; (2) o ego-mask do carrinho
não teria sido aplicado na run "~10:38", pois uma label descreveu o próprio veículo.

Comparação conceitual com a referência V1 (`/home/alexmrtr/Projects/contextual-3d-mapping`, o
"3d-map"), que pinta **ponto a ponto** (`projector.py`: `associate_map_points` atribui cor +
`region_id` + `label` a cada ponto). O ContextMap2 inverteu o modelo: `sensor_association`
produz `SpatialObservation` — cada região 2D guarda **referências** à geometria visível
(`geometry_support` = índices `geometry_id`), sem copiar cor nem label.

## Registro (AGENTS.md §22)

- **Problema.** Confirmar se a associação/projeção está correta e localizar a causa da "perda de
  cor" e do label de veículo.
- **Hipótese.** (H1) Projeção e pertencimento estão corretos; a "perda de cor" é de mecanismo de
  apresentação (índice de voxel), não de associação. (H2) A label de veículo vem da ausência de
  ego-mask: o carrinho é segmentado como foreground e rotulado.
- **Input.** Bundle congelado `workspace/frame-10_29-full-run` (frame único
  `camera_1_image_raw-000000`, câmera MEI 640×480, mapa bruto de 426.576 pontos, 15 scans).
  Reuso dos artefatos já produzidos — **sem re-execução** (respeita parâmetros congelados).
- **Processamento / métrica / critério.** Dois testes independentes, abaixo.
- **Output.** Veredito (H1/H2) e quantificação da contaminação do ego.

## Teste A — pertencimento (`validate_membership.py`)

Para cada região, os pontos listados em `geometry-support.u32` têm seu pixel projetado
(`prepared_u/v`, de `debug/.../samples.csv`) arredondado (`floor(coord+0.5)`) e testado contra a
máscara `.npy` da região (bit-unpacked, 480×640).

- **Critério:** ≥ 99,5% dos pontos pintados dentro da máscara.
- **Resultado: 62.067 / 62.067 = 100,00%**, todos em estado `associated`.
  → pertencimento, mapeamento `geometry_id`→ponto, arredondamento e espaço de coordenadas da
  máscara estão corretos.

## Teste B — reprojeção independente (`reproject_independent.py`)

Reimplementa do zero (numpy puro, sem reusar o código do pipeline) o modelo MEI (esfera + `xi` +
distorção radial-tangencial + intrínsecos), o slerp da pose `T_map_epson` e a composição da
extrínseca `T_epson_d`; projeta as coords 3D de `geometry.bin` e compara com os pixels gravados.

- **Critério:** erro de reprojeção mediano < 0,5 px.
- **Resultado: erro = 0,0000 px** (mediana, p99 e máx) nos 223.734 pontos in-support; 0 pontos
  acima de 0,5 px.
  → a cadeia pose → extrínseca → MEI → pixel reproduz o pipeline bit-a-bit. **Não há bug de
  projeção.**

## Contaminação do ego (H2 confirmada)

`state_counts.outside_valid_support = 0` e `normalization_rejections` só com `merged_duplicate`/
`region_budget_exceeded` (nenhuma `OUTSIDE_VALID_REGION`): **nenhuma exclusão foi aplicada**. O
carrinho é segmentado como foreground e pintado. Pontos pintados com labels do próprio veículo:

| região | label | pontos |
|---|---|---:|
| region-0005 | vehicle | 2.790 |
| region-0002 | robotic vehicle | 1.936 |
| region-0011 | off-road vehicle | 397 |
| region-0072 | car dashboard | 183 |
| region-0090 | tire | 131 |
| region-0018 | blurred wheel | 100 |
| region-0004 | all-terrain vehicle | 47 |
| region-0037 | blurred wheel | 20 |
| **total** | | **~5.604** |

Propagam-se ao artefato final: `entity-0001 = "robotic vehicle"` (e "all-terrain vehicle",
"blurred wheel", etc.) em `context_map/entities/entities.jsonl`.

## Veredito

- **H1 confirmada:** projeção e pertencimento estão corretos (0 px, 100%). A "perda de cor" do
  viewer **não** é bug de associação — é o índice posicional (`geometry_id` = índice na nuvem
  bruta) que não sobrevive à apresentação por centroide de voxel (ver
  `src/contextmap/spatial_support/docs/README.md`). → tratar na **Fase 3** (spatial_support →
  viewer).
- **H2 confirmada:** não existe ego-mask no ContextMap2; ~5.600 pontos recebem labels do próprio
  carrinho. → tratar na **Fase 2** (ego-mask via `ExclusionRegion`).

## Fase 2 (offline) — efeito da ego-mask + valid_region

`ego_mask_offline_effect.py` aplica as duas regras de normalização às 95 regiões já
descobertas do frame-10_29, usando os assets reais (banda inferior da V1 para o ego, círculo
fisheye para o válido), **sem re-rodar SAM/Qwen**. Assets convertidos para `.npy` em
`examples/v0.2.0/assets/corridor-02/`.

Com os thresholds **default** (`minimum_valid_fraction=1.0`, `maximum_exclusion_fraction=0.0`):

- **8/8 regiões rotuladas como veículo removidas, 0 sobrevivem** (elimina `entity-0001`
  "robotic vehicle" e as demais);
- mas **16 rejeições colaterais** de estrutura/chão — incluindo `region-0001` "curved wall" (a
  dominante, 46.880 pts), que apenas encosta na vinheta e é 100%-rejeitada por `valid<1.0`.

Varredura de thresholds (ego removidos / colaterais estrutura-chão):

| `exclusion>` | `valid<` | veículo removidos | colaterais |
|---:|---:|---:|---:|
| 0.0 | 1.0 | 8/8 | 16 |
| 0.0 | 0.5 | 8/8 | 7 |
| 0.5 | 0.5 | 4/8 | 7 |

**Decisão:** `minimum_valid_fraction = 0.5` (rejeita só quando a maioria da região cai fora do
círculo — poupa a parede que tangencia a vinheta) e `maximum_exclusion_fraction = 0.0` (rejeita
a região que encostar no rig — necessário porque region-0002/0005 só o tangenciam). Resultado:
**8/8 veículo removidos, 7 colaterais** (chão/superfície tocando a borda do rig: `wooden floor`,
`sand`, `metal frame`, ...). Os 7 colaterais são o preço de reusar a **banda** da V1 em vez de
uma **silhueta ajustada** do rig; várias delas ("sand", "flooded hallway", "blurred texture")
são leituras ruins do chão escuro, e o chão é redundante em outras regiões.

Wiring: `[image_preparation."<calibration_id>"]` no config (overlay opt-in
`examples/v0.2.0/corridor-02-image-masks.toml`, **sem tocar no canonical congelado**), decodificado
em `runtime/image_masks.py` e injetado nas duas etapas de preparação
(`VisualPerceptionExecutor` e `SensorAssociationExecutor`). Validação end-to-end pendente de uma
re-run com GPU (SAM2+Qwen), fora do escopo desta análise offline.

## Reproduzir

```bash
# do repo root; python 3.12 com numpy (mise 3.12.14)
python experiments/rgb-lidar-association-validation-20260930/validate_membership.py
python experiments/rgb-lidar-association-validation-20260930/reproject_independent.py
```

Saída de referência em `results.txt`.
