# pointcloud-raw-vs-voxel-20260930

Gerar **dois `map_id` independentes e completos** — `raw` e `voxel` — do corridor-02, para ver no
viewer o efeito da agregação por voxel direto na nuvem de pontos, **sem** rodar nenhuma etapa
semântica (percepção, fusão, mapeamento semântico, resolução de entidades, relações).

- **Controle:** `corridor-02-raw` — `GeometricMapArtifact` cru (todos os pontos medidos).
- **Teste:** `corridor-02-voxel` — 2º `GeometricMapArtifact` agregado por `ScanVoxelPolicy`
  (voxel-centroide intra-scan) no mesmo `cell_m`.

Os dois mapas saem de **um único `GeometryInputPlan`** (mesma sequência, mesma trajetória, mesmas
políticas) — só a agregação muda. Cada um é embrulhado num `ContextMap` só-geometria (0 entidades,
0 relações), o único artefato que o viewer cataloga e o menor que ainda renderiza geometria.

## Por que precisou de um driver de ingestão

O `contextmap ingest` da CLI **não carrega calibração externa**, e a mapeação geométrica precisa do
extrínseco `T_body_source` (`epson -> cmu_rc1_velodyne`, do `corridor-02-extrinsics.yaml`). O
`SequenceArtifact` canônico do corridor-02 foi apagado com `outputs/`. Então `ingest_corridor02.py`
reconstrói o mínimo para uma nuvem: ingere **só o LiDAR** (`/velodyne_points`) com a calibração dos
YAML, e a pose GT (`corridor-02-gt.txt`) como sequência auxiliar. Tudo no relógio `corridor-02-header`;
LiDAR e pose já compartilham essa época, então nenhuma correção de offset é necessária.

## Pré-requisitos

- Dataset em `datasets/corridor-02/` (`corridor-02/corridor02.bag`, `corridor-02-gt.txt`,
  `corridor-02-extrinsics.yaml`).
- Python 3.11+ com as deps do produtor **e `rosbags`** instalado (`pip install rosbags`, extra `ros1`).
- Ambiente do produtor: rodar via `PYTHONPATH=src` + o Python do mise 3.12.
  (Atalho neste checkout: `MP="$HOME/.local/share/mise/installs/python/3.12.14/bin/python"`.)

## Passos

Defina `B=experiments/pointcloud-raw-vs-voxel-20260930` e `OUT=$B/outputs/full`.

1. **Ingestão** (LiDAR + pose). Omita `--window-s` para o bag inteiro; use, ex., `--window-s 60 90`
   para um smoke de 30 s.
   ```bash
   PYTHONPATH=src "$MP" "$B/scripts/ingest_corridor02.py" \
       --dataset datasets/corridor-02 --output-root "$OUT"
   # imprime SEQUENCE_DIR=... e POSE_DIR=...
   ```

2. **Trajetória** (state_estimation por pose GT — external_pose, em CPU):
   ```bash
   SEQ=$(ls -d "$OUT"/sequences/corridor-02/*/); POSE=$(ls -d "$OUT"/sequences/corridor-02-pose/*/)
   PYTHONPATH=src "$MP" "$B/scripts/run_trajectory.py" \
       --sequence-dir "$SEQ" --pose-dir "$POSE" --workspace "$OUT" -c "$B/config.toml"
   # escreve $OUT/state_estimation
   ```

3. **Dois mapas** (raw + voxel), cada um num context_map só-geometria. Escreve **direto no
   `workspace/`** que o viewer varre, num nível que o deep-scan alcança (`if (depth < 2)`):
   ```bash
   PYTHONPATH=src "$MP" "$B/scripts/build_pointcloud_maps.py" \
       --sequence "$SEQ" --trajectory "$OUT/state_estimation" \
       --output-dir workspace/pointcloud-raw-vs-voxel-20260930 \
       --map-name corridor-02 --cell-m 0.10 -c "$B/config.toml"
   ```
   Layout produzido (o locator de geometria resolve para `../geometric_mapping`):
   ```text
   workspace/pointcloud-raw-vs-voxel-20260930/corridor-02-raw/{geometric_mapping,context_map}
   workspace/pointcloud-raw-vs-voxel-20260930/corridor-02-voxel/{geometric_mapping,context_map}
   ```
   As sequências e a trajetória ficam em `$OUT` (intermediários); só os mapas vão para `workspace/`.

## Ver no viewer

O `.env` do `contextmap2-viewer` já aponta `NUXT_CONTEXT_MAP_ARTIFACTS_ROOT` para
`.../contextmap2/workspace` (git-ignored). Basta subir com `npm run preview`: o deep-scan
**descobre automaticamente os dois cards** (`corridor-02-raw`, `corridor-02-voxel`) — nenhuma
mudança de env. Cada um renderiza como nuvem normal, controle e teste lado a lado. Não é preciso
`NUXT_VOXEL_ARTIFACTS_ROOT`: aqui o voxel é a própria geometria do mapa-voxel (intra-scan), não uma
fonte inter-scan.

> Profundidade: o scan desce só ~3 níveis a partir da raiz, então os mapas têm de ficar rasos em
> `workspace/` (ex.: `workspace/<grupo>/<map>/context_map`), não aninhados fundo em `outputs/`.

## Validação (smoke de 30 s)

`--window-s 60 90` produziu: raw = 2.239.629 pontos; voxel (0.1 m) = 417.360 pontos (−81%); ambos os
context_maps `entity_count=0`, `relation_count=0`, validados em `ValidationLevel.FULL`, locator
resolvendo. A cadeia ingest → state_estimation → mapas roda ponta a ponta em CPU.

## Fora de escopo

- Voxel inter-scan (`VoxelAggregationArtifact`) como card próprio: o viewer só o mostra como fonte
  alternativa (toggle) de um mapa, não como `map_id`.
- Câmera/IMU e correção de offset: desnecessários para uma nuvem só-geometria; por isso não entram.
- Comando CLI de 1ª classe para mapa só-geometria: adiado (YAGNI).
