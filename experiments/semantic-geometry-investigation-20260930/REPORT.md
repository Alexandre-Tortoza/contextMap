# Investigação: a semântica altera a geometria? Projeção, máscaras e explosão de labels

Run investigado: `workspace-old/frame-10_31-{raw,voxel}` (corridor-02, imagem
`camera_1_image_raw-000055`, 10:31), produzido por
`experiments/frame-10_31-raw-vs-voxel-20260930/run_two_arms.py`. É o run cujas labels
coincidem exatamente com a lista reportada (29 labels no braço voxel, 30 no raw).
Nenhum código de produção foi alterado. Todos os números abaixo foram medidos pelos scripts
em `scripts/` sobre os artefatos imutáveis; o que não foi medido está marcado como hipótese.

Convenção: **medido** = número produzido aqui; **verificado no código** = leitura de
arquivo:linha; **hipótese** = mecanismo plausível ainda não isolado.

---

## 1. Pipeline real atual (o que o run executou)

```text
bag ROS1 corridor02.bag + corridor-02-gt.txt + YAMLs de calibração
│ ingestion (ros1_bag adapter; ingest_window_fresh.py)
│   janela 628–637 s (631 s ± padding 3 s); correção de época constant_offset
│   78621358412432135 ns aplicada IGUAL a câmera, LiDAR, IMU e GT (mesmo relógio "header")
│   calibração reconstruída dos YAMLs: T_epson_velodyne (laser_to_imu), T_epson_d
│   (rgb_camera_to_imu), câmera MEI 640x480 (xi=1,563; k1,k2,p1,p2; gamma1/2; u0/v0)
│   saída: SequenceArtifact (216 imagens, 89 scans, 1798 IMU), metros, ns
▼
state_estimation external_pose (GT TUM, xyzw) → T_map_epson(t), 1 pose por amostra GT
│   (gaps entre amostras: 101 ms ×66, 202 ms ×47, até 605 ms perto da imagem)
▼
geometric_mapping (GeometricMappingExecutor)
│   por scan: P_map = T_map_epson(t_header) · T_epson_velodyne · P_velodyne   (uma vez)
│   t_header = INÍCIO do scan (campo time por ponto: 0 → 100,8 ms); SEM deskew
│   ("No corrector exists", motion_correction.py; policy unknown=warn)
│   pose por interpolação, gap máx 400 ms → 8 scans rejeitados (55–59, 76–78)
│   raw: 81 scans → 2.322.324 pts | voxel: ScanVoxelPolicy 0,10 m POR SCAN → 402.626 pts
│   geometry.bin: <3d3dIqI> map xyz | source xyz | scan | índice no scan | count
▼
visual_perception (1 frame: imagem 55; preset canonical/1 SEM scene_interpretation)
│   image_preparation (valid_region + exclusão ego, overlay corridor-02-image-masks.toml)
│   SAM2.1-hiera-tiny AutomaticMaskGenerator: pred_iou_thresh=0, stability_thresh=0
│     → 431 propostas → orçamento 256 (top IoU predito) → normalização: 113 merges
│       (99 contenção ≥0,95 absorvidos pelo MAIOR, 14 IoU ≥0,8), 44 exclusão ego,
│       16 fora da região válida → 83 regiões aceitas
│   DINOv2 denso, CLIP por região
│   Qwen3-VL-4B 4-bit, prompt region-label/v1, view tight_crop (bbox, SEM máscara)
│     → 77 claims (1 primária por região, 0 alternativas), 6 falhas de parse
▼
semantic_scoring: CLIP cos("a photo of a {label}", recorte) → 0,13–0,27 (cru)
▼
sensor_association (FrameProjector.project, frame_projection.py:444)
│   T_map_d(t_img) = T_map_epson(t_img) ∘ T_epson_d   (interpolado, fração 0,166, bracket 302 ms)
│   candidatos = TODO o mapa (81 scans, 2,32 M pts), P_cam = (P_map − t)·R (linha 509)
│   MEI → pixel; suporte válido; oclusão: célula 4 px, janela 5×5 células, margem
│   max(0,1 m; 2 %) sobre o alcance do raio (visibility.py:254–295)
│   pertencimento: ponto visível ∈ TODA máscara que contém o pixel (membership.py:354)
│   saída: 83 SpatialObservation (referências u32 ao mapa; nenhum XYZ)
▼
semantic_fusion: supports = componentes conexos de Jaccard ≥ 0,3 entre conjuntos de pontos;
│   hipóteses agrupadas por texto NFKC+casefold (accumulation.py:191) → 41 supports
▼
semantic_mapping: one-support-one-entity-v1 → 41 entidades; primária = hipótese única;
│   min_supporting_physical_observations = null; atributos = só "class" (state_mapping.py:140)
▼
entity_resolution: só canal GEOMETRY (jaccard 0,5 / IoU 0,5 / contenção 0,9) → 38 resolvidas
▼
spatial_relations → context_map: referencia o geometric_map por locator + map_id + point_count;
│   entidades = listas de geometry_id (índices) + hipóteses {label}
▼
viewer (fora da Solução 1): desenha o geometry.bin; com artefato de associação acessível,
    pinta de 0x333333 a 40 % de opacidade todo ponto sem observação espacial
    (GeometryChunks.client.vue:86–87, 257–258; evidence-adapter.ts:86)
```

Frames e unidades em todo o caminho: metros, nanossegundos, frame `map` (Z para cima).
Contagem de pontos por estágio no frame 55 (braço raw): mapa 2.322.324 → projetáveis
1.783.217 → na imagem 1.549.066 → no suporte válido 1.540.099 → visíveis 23.948 →
associados 21.602 (0,93 %) → em supports 21.594 → em entidades do context_map 21.594.

## 2. Pipeline relevante do V1 (`contextual-3d-mapping`)

Verificado no código (`apps/mapping-runtime/src/mapping_runtime/corridor02_context.py`,
`modules/sensor-association/src/sensor_association/projector.py`,
`modules/visual-perception/src/visual_perception/application/semantic_grounding.py`) e nos
manifests dos runs `artifacts/runs/2026-09-18-run-00{1,2,3}-corridor-02-objects-*`:

```text
keyframe RGB ──► _nearest_lidar: scan LiDAR com header mais próximo (≤100 ms)
                 │ associação no frame LOCAL do LiDAR:
                 │   P_cam = T_cam_lidar(REFINADA) · P_velodyne     (a pose NÃO entra no pixel)
                 │   T_cam_lidar = Rcorr(yaw 4,25°, pitch −2,25°) · inv(camera_to_imu) · laser_to_imu
                 │   (manifest corridor-02-camera-lidar-calibration-v2.json, issue #285)
                 │ MEI (mesma fórmula), z-buffer por pixel exato (1 ponto/pixel)
                 │ regiões: footprints DISJUNTOS (projector._region_index recusa sobreposição)
                 │   gerados por grounding: label → GroundingDINO → SAM2.1-hiera-large;
                 │   posse do pixel por ordem (menor área primeiro); margem de borda
                 │   → ponto a ≤ margem da borda recebe só tentative_label
                 │ VLM: gemini-robotics-er-2-preview; claims tipados (label, attribute,
                 │   material, condition, hazard, host_surface), superfícies estruturais
                 │   movidas para structural_context
                 ▼
nuvem semântica = os pontos DESSE scan local (stride), transformados para o mapa só p/ exibir;
o PCD global é apenas backdrop, NUNCA candidato à associação
```

## 3. Diferenças conceituais de data flow (V1 → atual)

| aspecto | V1 | ContextMap2 (run 10:31) | evidência |
|---|---|---|---|
| calibração câmera↔corpo | base + correção yaw 4,25°, pitch −2,25° | **base do dataset, sem correção** | calibration.json da sequência; manifest v2 do V1 |
| geometria associada | 1 scan local sincronizado (≤100 ms) | **mapa acumulado inteiro (81 scans, −2,2 s…+6,6 s)** | frame_projection.py:444–509; summary.json `candidate_fraction: 1.0` |
| pose no pixel | não entra (frame do LiDAR) | T_map_epson interpolado no instante da imagem | frame_projection.py |
| deskew | irrelevante (scan local ~ mesmo instante) | **nenhum**; header = início do scan | motion_correction.py docstring; campo `time` 0–100,8 ms |
| visibilidade | z-buffer por pixel exato | janela 20×20 px com margem 0,1 m/2 % | visibility.py:254 |
| máscaras | disjuntas, por grounding do conceito, SAM2 large | **sobrepostas**, AMG sem filtro de qualidade, SAM2 tiny | membership.py:1–10; sam2.py:72–73 |
| ponto em várias máscaras | impossível | **92,3 % dos pontos associados em ≥2 máscaras (até 7)** | medido |
| entrada do VLM | footprint aterrado | recorte do bbox, máscara não marcada | semantic_views.py:375; view_policy tight_crop |
| estrutura semântica | claims tipados por slot | category/attributes existem no claim mas **não chegam à entidade** | state_mapping.py:140; accumulation.py:191 |
| nuvem pintada | pontos do scan local (novos IDs por keyframe) | pontos do mapa por índice (IDs persistentes) | — |

## 4. Primeira etapa em que a geometria diverge entre semantics OFF e ON

**Nenhuma. Medido.**

- Teste E (`scripts/p2_test_e_semantics_off.py`): o `GeometricMapArtifact` reconstruído
  das mesmas entradas **sem nenhum estágio semântico** tem `geometry.bin` byte a byte idêntico
  ao referenciado pelo context_map, nos dois braços:
  raw `sha256:d4e4b03e…`, voxel `sha256:b243d227…`; mesmo `map_id`, mesma contagem.
- Snapshots 01→08 (`results/p2_geometry_snapshots.json`):
  - 01 payload bruto do scan 22 (28.709 pts) == `source xyz` do geometry.bin: delta 0,0 exato.
  - 03 recomputação independente velodyne→epson→map (lerp+slerp próprios): máx 1,8e−13 m.
  - 06 associação: só referências u32 (53.977 refs, 21.602 pontos únicos); nenhum campo XYZ.
  - 07 fusão/entidades: centroides e bounds gravados == recomputados do geometry.bin
    (máx 2,6e−13 m); pontos da fusão ⊂ pontos associados.
  - 08 context_map: `geometry-reference.json = {map_id, point_count: 2322324}` + locator
    `../geometric_mapping`; pontos das entidades ⊂ associados.
- Deltas de geometria semantics OFF vs ON: contagem 0, ocupação de voxel 0, centroide 0,
  bbox 0, RMSE 0 (é o mesmo arquivo).

**Então por que a nuvem parece mudar?** Três mecanismos medidos, nenhum na semântica:

1. **Regra de cobertura do viewer** (verificado no código + medido): quando o viewer encontra o
   artefato de associação, todo ponto sem observação espacial vira `0x333333` com opacidade
   0,4. No frame 55 isso é **99,07 %** do mapa raw (98,97 % no voxel). Os mesmos XYZ passam a
   parecer outra nuvem — menos pontos, outra distribuição, outra densidade
   (`figures/f08_viewer_coverage_effect_topdown.png`, painéis b vs c).
2. **Outro mapa geométrico de entrada**: os mapas só-geometria em disco são
   `frame-10_28-raw` (**1 scan, 28.500 pts**) e `corridor-02-raw` (bag inteiro, 67,1 M). O run
   semântico usa um mapa de **81 scans em 8,9 s (2,32 M pts)**, determinado pela janela de
   ingestão, não pela semântica (f08, painel a vs b).
3. **Fantasmas do acúmulo sem deskew** (medido, parte 5): no mapa de 81 scans a distância
   ponto-plano entre scans vizinhos (≤3 s) tem mediana 3,3 cm / p90 10,0 cm; com deskew por
   ponto cai para 2,2 / 6,9 cm. Paredes "grossas" e bordas duplicadas existem no mapa de
   entrada, sem ou com semântica.

Não observei a tela do usuário: qual dos três (ou combinação) produziu a percepção é
hipótese. Como provar: abrir o mesmo context_map com e sem `NUXT_UPSTREAM_ARTIFACTS_ROOT`
(sem ele a cobertura retorna 204 e nada é esmaecido) e comparar com o card só-geometria.

## 5. Auditoria de transforms e timestamps

| transform | origem | convenção | aplicado em | verificação |
|---|---|---|---|---|
| T_epson_velodyne | YAML `laser_to_imu` (4×4) → quatérnio xyzw [0, 0,00475, 0, 0,99999], t = 0 | T_parent_child, p_parent = R·p_child + t, rotação ativa | geometric_mapping, 1× por scan | recomputado: 1,8e−13 m; sentido **não discriminável** (0,54°; resíduo multi-scan 2,161 vs 2,201 cm invertido) |
| T_epson_d | YAML `rgb_camera_to_imu` → xyzw [0,5026, −0,5028, 0,5050, −0,4893], t = [0,173; 0,048; −0,019] m | idem; frame `d` é óptico (z→+x corpo, x→−y, y→−z) | sensor_association, 1× por frame | eixos ópticos corretos; **rotação desalinhada ~4° (seção 6)** |
| T_map_epson(t) | GT TUM `t tx ty tz qx qy qz qw` | xyzw, metros, mesmo relógio corrigido | mapa (t = header do scan) e associação (t = imagem) | offset GT↔sensor: ótimo em **0…+25 ms** com deskew (p90 6,85 cm em 0) |
| P_cam | `(P − t)·R` em vetores-linha = Rᵀ(P − t) | inversa correta de T_map_d | frame_projection.py:509 | teste A: ida/volta < 1e−9 m |

Checagens explícitas: sem transform aplicada duas vezes; sem T_world_cam × T_cam_world trocado;
extrínseca da câmera não invertida (uma inversão faria a câmera olhar para outra direção);
`d` é frame óptico, não base_link; quatérnio xyzw em todo lugar (TUM também); metros;
matrizes 4×4; rotação ativa. Nenhuma associação por índice: câmera↔pose por timestamp.

Timestamps do frame 55: `rgb = 1724621714,984856033`; scan mais próximo
`velodyne_points-000022` começa −50,26 ms antes (cobre −50 → +50 ms: centrado na imagem);
pose GT: bracket `pose-004223 → 004224`, gap 302,6 ms, fração 0,166, Δ até a amostra mais
próxima 50,26 ms. A correção de época é um offset constante único para câmera, LiDAR, IMU e
GT (medido: bag−evento, mediana −25,0 ms imagem / −27,6 ms LiDAR). Plataforma no instante:
2,1 m/s, 4,2 °/s.

## 6. Resultado da projeção 3D→2D

- Modelo: **MEI/CamOdoCal** correto (xi, k1, k2, p1, p2, gamma1/2); **não** é pinhole
  ingênuo. Domínio: `z > (−min(xi,1/xi)+1e−9)·alcance`; sem dobra de distorção.
- Implementação: reimplementação independente (numpy) reproduz o artefato **exatamente**
  (0/83 regiões divergentes, 5 contagens de estado idênticas, nos dois braços; teste C:
  MEI do pipeline = fórmula independente < 1e−6 px; projeção→raio→XYZ < 1e−6 m).
- **Calibração: desalinhada.** Replicando o critério do V1 (consistência fotométrica
  multi-frame) nos dados do próprio ContextMap2 (GT, deskew, 7 trechos de 8 frames):
  melhor ajuste **yaw +4,0°, pitch −1,75°**; por trecho, yaw 3,0–4,5°, pitch −0,25…−2,75.
  Inconsistência: ajuste 0,204 → 0,136; **validação 0,192 → 0,132 (−31 %)** com a correção
  do V1 (4,25°/−2,25°), igual ao melhor ajuste (0,133). Deslocamento resultante dos pontos
  visíveis: **mediana 24,4 px** (p10 22,2; p90 25,4).
- Percentuais no frame 55 (raw): atrás da câmera 23,21 %, fora da imagem 10,08 %, fora do
  suporte válido 0,39 %, ocluídos 65,29 %, visíveis 1,03 %.
- Visual: `figures/f02_*` (scan local, base vs corrigido), `f03_*` (visíveis do pipeline),
  `f06_*` (pontos por máscara, vermelho = base, verde = corrigido).
- Efeito na contaminação por camadas de profundidade (fração ponderada de pontos fora da
  camada principal da máscara): scan local + z-buffer por pixel **14,6 % → 7,2 %**;
  scan local + oclusão do pipeline 2,3 % → 0,8 %; mapa acumulado + oclusão do pipeline
  0,42 % → 0,24 %; mapa acumulado + z-buffer por pixel 1,7 % → 2,1 % (não melhora: o fundo
  vaza pelos buracos do mapa esparso). Regiões finas: janela 0020 75 % → 0,3 %; batente 0062
  79 % → 0 %; corrimão 0035 28 % → 0,7 %.
- Teste D (`results/test_d_known_pixel.json`): ponto da barra da porta esquerda
  (índice 481273, scan 16, 1,04 m): base → pixel (120,5; 76,3) ∈ {flooded hallway, metal
  handle, door}; corrigido → (142,1; 88,6), 24,9 px adiante, ∈ {flooded hallway, metal rod}.

**Visibilidade/oclusão** (seção pedida na parte 5): a regra do pipeline marca 98,4 % dos
pontos no suporte como ocluídos. Anatomia (medido): 77 % estão >1 m atrás da superfície de
suporte (oclusão real — o mapa cobre 77×88 m e a câmera vê um trecho); mas **12 %
(~185 mil) estão a ≤0,5 m** da profundidade de suporte — tipicamente a mesma superfície em
ângulo rasante ou engrossada pelos fantasmas, descartada pela janela de 20×20 px. Isso deixa
a pintura esparsa e "fragmentada".

## 7. Resultado da análise das máscaras (antes de qualquer label)

SAM2 re-executado no frame 55 com a mesma configuração (determinístico: 431 propostas; as 83
regiões aceitas casam com IoU ≥ 0,99): `results/p6_sam2_rerun.json`, `p6_mask_quality.json`,
`figures/f04_masks_numbered.png`, `f05_mask_overlap_count.png`.

- IoU predito: mediana 0,30 (p10 0,027); estabilidade: mediana 0,72.
- **Só 10/431 propostas e 2/83 regiões aceitas passariam nos filtros padrão do próprio SAM2**
  (0,8/0,95). O backend define 0,0/0,0 por default (sam2.py:72–73, desde #235) e nenhum
  config os sobrescreve.
- 83 máscaras, mediana 1.100 px; 23 < 500 px; 27 com IoU predito < 0,5; 47 com estabilidade < 0,8.
- Fragmentação: 68,7 % com >1 componente conexo; 20 com >10; as maiores são salpicadas
  (0010 "blue door": 394 componentes, estab. 0,16; 0015: 239; 0069: 195; 0001: 171).
- Sobreposição: 43/83 máscaras ≥80 % dentro de outra; profundidade de aninhamento por pixel:
  só 15 % dos pixels cobertos têm 1 máscara (2: 39 %, 3: 36 %, até 7).
- Merge por contenção ≥0,95 elege o **maior** como representante (normalization.py:56–58):
  partes pequenas totalmente contidas (maçaneta, trinco) somem como região própria.
- Casos da lista: parede em várias regiões (sim, 0005/0017/0022 sobrepostas), mesma porta com
  várias máscaras (0001/0069/0070 e 0010/0015), porta incluindo parede (0017 cobre porta +
  parede), fragmentos pequenos mantidos (23 < 500 px), fundo como objeto (0005 com bbox de
  57 % da imagem → "flooded hallway").

**Conclusão por evidência:** o problema já existe nas máscaras, antes de qualquer label.

## 8. Origem da explosão de labels

Tabela completa por claim: `results/p7_label_provenance.csv` (frame, máscara, label,
category, attributes, region_kind, CLIP, IoU/estabilidade SAM2, bbox, support, entidade,
modelo/revisão, template/fingerprint do prompt, resposta bruta).

- Fonte única: **VLM Qwen/Qwen3-VL-4B-Instruct** (rev. `ebb281ec…`, 4-bit, T=0),
  template `region-label/v1` (não canônico), view `tight_crop`. CLIP só pontua (cosseno cru
  0,13–0,27, não é probabilidade). Não há Detic/GroundingDINO/tagger no run.
- O VLM recebe o **bbox sem a máscara marcada** (a opção `masked_subject`/`contextual_crop`
  existe e não foi usada): para máscaras salpicadas o bbox é quase a imagem inteira e o modelo
  nomeia o objeto mais saliente ou a cena. Exemplos: 0005 → "flooded hallway" (descrição: "a
  long, empty hallway with water on the floor…"); 0022 (28.971 px) → "metal pole"; 0070
  (máscara porosa sobre a porta) → "door handle"; **0019 → "spacecraft"** ("curved edge of a
  spacecraft…": recorte da borda da vinheta fisheye com uma haste; IoU 0,88/estab. 0,91 — a
  máscara é boa, a alucinação é do VLM sobre um recorte sem contexto).
- 77 claims, **1 por máscara**, 0 alternativas (o prompt pede alternativas), 6 falhas de
  parse (`claims` ausente) → 2 entidades `insufficient_evidence`.
- **Descrição virando classe:** 42/77 labels trazem cor/material no próprio texto; só 2 delas
  repetem o valor em `attributes`. O claim tem `category` (14 valores; "door" em door /
  structure / thing) e `attributes` (77/77), mas a fusão agrupa só pelo texto e o estado da
  entidade expõe apenas `class = label`: cor, material e categoria **não chegam ao mapa**
  (state_mapping.py:140). Problema arquitetural: o mapa é uma lista plana de strings.
  `region_kind` saiu "thing" em 76/77 (inclusive paredes).

## 9. Redundância e promoção de observações a entidades

`results/p9_label_redundancy.json`, `p10_entity_chain_and_overlap.json`.

- Similaridade (encoder de texto do CLIP do scorer): cosseno fora da diagonal mediana 0,65;
  pares mais próximos: door↔doorway 0,95, window↔window frame 0,92, metal door↔door 0,90,
  metal rail↔metal railing 0,89, metal rod↔metal pole 0,86. 38 labels → 32 clusters
  (ligação média, cos ≥ 0,85) ou 26 (núcleo léxico). Entropia: 4,90 bits (máx 5,25).
- Mecanismos: **A** (mesma máscara com várias labels) = **0**. **B** = 9 supports misturam 24
  labels distintas — máscaras sobrepostas de granularidade diferente (inteiro/parte) viram um
  support por Jaccard ≥ 0,3 (ex.: entity-0027 = door + flooded hallway + metal pole + metal
  railing). **C** (multi-frame) não se aplica a este run de 1 frame. **D** = sim: a chave é o
  texto exato; door/blue door/metal door são hipóteses diferentes e cada string vira um item
  da lista de labels do mapa.
- As labels parecidas são, na maior parte, **objetos diferentes**: door/blue door/metal door
  aparecem em 12 regiões e só 7 de 66 pares se sobrepõem na imagem; metal pole/metal rod em 10
  regiões, 1 par sobreposto. Em 3D, as 7 entidades da família "door" formam ~3–4 grupos
  (0017 ⊂ 0027 a 97,6 % dos voxels; 0025/0026 68,5 %; 0011~0032 64 %; 0002 isolada) — contagem
  aproximada, sem verdade de campo.
- Promoção: 83 regiões → 83 observações (8 sem ponto, 23 com <5 pontos excluídas) → **41
  supports = 41 entidades** (`one-support-one-entity-v1`, 0 rejeitadas) → 38 resolvidas → 38
  no mapa, **todas com 1 observação física**, 27 com primária "unambiguous". Ou seja: uma
  máscara de um frame vira entidade persistente com label primária
  (`min_supporting_physical_observations: null`; a política `MinimumSupportPrimaryPolicy`
  existe e não foi configurada).
- Multi-view (não medido neste run; medido antes em `relation-graph-diagnostic-20260929`,
  run canonical-2 de 4 frames): 606 entidades → 535 resolvidas, 90,5 % singletons, 1,13 fontes
  por entidade; só o canal de geometria é avaliado na resolução. Critérios hoje: Jaccard/IoU/
  contenção de pontos 3D; aparência, semântica e temporal ficam `not_evaluated`.

## 10. Hipóteses de causa raiz, classificadas por evidência

| # | hipótese | evidência | arquivo:linha | reproduzir | provar/refutar |
|---|---|---|---|---|---|
| H1 | **Extrínseca câmera↔corpo ~4° yaw / ~2° pitch errada** desloca a projeção ~24 px | **medida** (fotométrico −31 % em validação; ajuste independente = correção do V1; f02/f06) | calibration.json; frame_projection.py:509 | `p4_calibration_photometric.py`, `p5_association_ablation.py` | já provada nesta janela; falta confirmar em outros trechos do bag |
| H2 | **Máscaras de baixa qualidade** (sem filtro SAM2) fragmentam e sobrepõem regiões | **medida** (2/83 passariam 0,8/0,95; 69 % fragmentadas; 85 % dos pixels com ≥2 máscaras) | sam2.py:72–73; normalization.py:55–58 | `p6_sam2_rerun.py`, `p6_mask_quality.py` | ablação com limiares explícitos, mesma imagem |
| H3 | **Pertencimento não exclusivo** pinta um ponto com até 7 regiões | **medida** (92,3 % em ≥2; 2,5 regiões/ponto) | membership.py:354 | `p5_association_ablation.py` | política de partição/hierarquia vs atual |
| H4 | **VLM vê bbox sem máscara** → labels de objeto saliente/cena e alucinação | medida em casos (0005, 0022, 0070, 0019) + código | semantic_views.py:375; config `tight_crop` | `p7_label_provenance.py`, f07 | rodar a mesma região com `masked_subject`/`contextual_crop` |
| H5 | **Texto livre como chave** e atributos descartados → explosão de labels | verificado no código + medido (42/77 com adjetivo no texto) | accumulation.py:191; state_mapping.py:140 | `p7`, `p9` | carregar category/attributes e medir labels únicas por entidade |
| H6 | **Promoção prematura**: 1 support → 1 entidade com primária, 1 frame | verificado no código + medido | materialization.py (one-support-one-entity-v1) | `p10_entity_overlap.py` | configurar min support e medir entidades/primárias |
| H7 | **Oclusão agressiva** descarta a mesma superfície → pintura esparsa | medida (12 % dos ocluídos a ≤0,5 m do suporte) | visibility.py:291 | `p5_occlusion_anatomy.py` | variar raio/margem e medir visíveis vs contaminação |
| H8 | **Viewer esmaece 99 %** dos pontos com cobertura → "nuvem muda" | verificado no código + contagem | GeometryChunks.client.vue:257; evidence-adapter.ts:86 | f08 | abrir o mesmo mapa com/sem upstream root |
| H9 | **Mapa de 81 scans sem deskew** tem fantasmas de ~7–10 cm (p90) | medida | motion_correction.py; geometric_mapping | `p3_time_offset_deskew.py` | implementar deskew e repetir |
| — | ~~semântica reconstrói/transforma XYZ~~ | **refutada** (teste E, snapshots) | — | `p2_*` | — |
| — | ~~projeção pinhole em câmera fisheye~~ | **refutada** (MEI correto) | camera_models.py:822 | teste C | — |
| — | ~~offset de relógio GT↔sensores~~ | **refutada** (ótimo 0…+25 ms com deskew) | — | `p3_time_offset_deskew.py` | — |

Qual delas domina a *aparência* da pintura no viewer é hipótese; H1–H3 são as de maior efeito
medido sobre quais pontos recebem qual máscara.

## 11. Correções propostas (não implementadas)

Ordem = prioridades pedidas. Nenhuma altera o cenário congelado sem evidência nova; cada uma
entra como overlay/experimento com uma variável por vez.

**C1 — Prioridade 1: esmaecimento por cobertura no viewer (H8)**
- Problema: o viewer torna 99 % da geometria cinza translúcida quando há evidência.
- Justificativa: a geometria deve continuar legível independente da semântica.
- Entrada: bitmap de cobertura (`evidence/coverage`). Processamento: tornar a cobertura uma
  camada opcional (desligada por padrão) com legenda, sem alterar a cor base.
- Saída: mesma nuvem opaca com/sem context_map. Métrica: fração opaca (hoje 0,93 %).
- Teste: teste de componente do viewer — sem a camada ligada, `displayedAppearance` não
  depende de `observationCoverage`. (Repositório do viewer, fora da Solução 1.)

**C2 — Prioridade 2: calibração câmera↔corpo refinada e versionada (H1)**
- Problema: T_epson_d desloca a projeção ~24 px.
- Justificativa: dois estimadores independentes (V1 com FAST-LIO, este com GT) convergem para
  ~4°/−2°; −31 % de inconsistência fotométrica em validação.
- Entrada: YAMLs + manifest de refinamento (yaw, pitch, roll, convenção, janela de estimação).
- Processamento: na ingestão, compor `Rcorr` com `rgb_camera_to_imu` gerando **novo
  calibration_id** e registrando a conversão em `provenance.conversions_applied` (sem editar
  os YAML). Estimador fica em `evaluation`/experimento.
- Saída: CalibrationSet com identidade nova. Métrica: inconsistência fotométrica em trechos
  de validação de pelo menos 3 pontos do bag; deslocamento em px; contaminação por camada.
- Teste: unitário da composição (convenção Rz·Rx·Ry no frame óptico, sentido do deslocamento
  em pixels) + a avaliação `p4` como gate.

**C3 — Prioridade 2: deskew por ponto no geometric_mapping (H9)**
- Problema: header = início do scan, nenhum corretor; fantasmas p90 10 cm.
- Entrada: scan com campo `time`, trajetória. Processamento: um corretor explícito
  (`MotionCorrectionState.CORRECTED` com evidência), pose por fatia de tempo.
- Saída: mapa com scans corrigidos. Métrica: ponto-plano multi-scan (mediana 3,3 → 2,2 cm;
  p90 10,0 → 6,9 cm medidos offline). Teste: scan sintético com plataforma em movimento
  reconstrói um plano com resíduo < ruído.

**C4 — Prioridade 3: candidatos por tempo e/ou oclusão calibrada (H7, H1)**
- Problema: todo o mapa é projetado e a janela 20×20 px descarta a própria superfície.
- Processamento (a avaliar, não decidido): (a) política de candidatos restrita aos scans a
  ±Δt da imagem (o geometry.bin já guarda o scan de cada ponto) — análogo ao V1 sem perder
  índices persistentes; (b) varrer `neighborhood_radius_cells` 0–2 e margens.
- Métrica: visíveis, contaminação por camada, fração ocluída a ≤0,5 m. Teste: cena sintética
  com duas superfícies (frente/trás) e uma parede rasante.

**C5 — Prioridade 3: política de pertencimento explícita (H3)**
- Problema: 2,5 regiões por ponto; um ponto pinta porta, parede e corredor ao mesmo tempo.
- Processamento: manter as sobreposições como evidência, mas adicionar uma política
  versionada (p.ex. posse do menor, como o V1, ou hierarquia parte-de) para a pintura e para a
  formação de supports. Métrica: regiões/ponto, supports com labels mistas (hoje 9/41).
  Teste: duas máscaras aninhadas → cada ponto em exatamente uma.

**C6 — Prioridade 4: qualidade das regiões (H2)**
- Problema: limiares SAM2 0/0. Processamento: declarar `predicted_iou_threshold`/
  `stability_threshold` no config e ablar (0/0, 0,5/0,8, 0,8/0,95) na mesma imagem; o default
  do SAM2 manteria só 10 propostas, então o valor precisa de avaliação, não de chute.
- Métrica: máscaras/frame, fragmentação, cobertura dos pixels úteis, aninhamento.
  Teste: o config efetivo registra os limiares; o digest muda com eles.

**C7 — Prioridade 4: view do VLM com a máscara (H4)**
- Processamento: `region_views = ["masked_subject"]` ou `contextual_crop` com contorno
  (já implementados). Métrica: labels de cena por região grande, taxa de alucinação em uma
  amostra anotada, CLIP do recorte mascarado. Teste: requisição contém a view mascarada.

**C8 — Prioridade 5: semântica estruturada sem ontologia rígida (H5)**
- Processamento: propagar `category` e `attributes` do claim até o estado da entidade
  (open vocabulary, evidência preservada); regra de refinamento versionada
  (`refinement_modifiers`, já existe em `entity_resolution/semantic_comparison.py`) só como
  canal de comparação, nunca reescrevendo a label original.
- Métrica: labels únicas/entidade, clusters por entidade. Teste: "blue door" + color=blue
  preserva ambos e compara igual a "door" no canal semântico.

**C9 — Prioridade 5: não promover observação única a verdade (H6)**
- Processamento: configurar `MinimumSupportPrimaryPolicy` (p.ex. ≥2 observações físicas) e
  avaliar os canais de aparência/semântica na resolução (calibração H5 em andamento).
- Métrica: entidades com primária, observações/entidade, singletons (90,5 % no run de 4
  frames). Teste: entidade de 1 frame mantém hipóteses mas `primary_hypothesis = null`.

---

## Reproduzir

Ver `README.md`. Resultados em `results/`, figuras em `figures/`, testes A–E em
`diagnostic_tests/` (5/5 passam).
