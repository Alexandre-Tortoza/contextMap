# Run de dois frames pelo CLI: corridor-02 10:31 (imagens 55 e 61), braços raw e voxel

Data: 30/09/2026. Tudo local, sem commit. Identidade de código em `code-identity.txt` (ver
"Identidades de código" abaixo).

## Objetivo

Produzir dois `ContextMapArtifact` completos, só pelo CLI `contextmap`, com o VLM dando as labels
e o fator de interação que o core converte em raio por objeto e na relação
`within_interaction_range`:

- frames `camera_1_image_raw-000055` (10:31) e `camera_1_image_raw-000061` (6 imagens depois,
  ~0,25 s), fundidos num mapa por braço;
- **raw**: mapa com todos os pontos (deskew); **voxel**: o mesmo mapa com agregação voxel
  intra-scan de 0,10 m;
- a percepção roda uma vez, no braço raw, e o braço voxel a **seleciona**: regiões e labels são
  as mesmas nos dois.

## Layout

```text
workspace/
├── run-frame-10_31/                 # run store (--workspace)
│   ├── sequences/corridor-02/frame-10_31-refined      # ingest câmera/LiDAR/IMU, calibração refinada
│   ├── sequences/corridor-02-pose/frame-10_31-gt      # ingest do GT (TUM)
│   └── corridor-02/
│       ├── run-0001   fundação raw (trajetória GT + mapa com deskew, 2.264.716 pts)
│       ├── run-0002   bloqueado no pré-voo (pose_ingestion fornecida sem uso), sem artefato
│       ├── run-0003   fundação voxel (mesma trajetória, mapa voxel 0,10 m, 393.054 pts)
│       ├── run-0004   context run raw, frame 55, prompt v1: descartado (fora das branches)
│       ├── run-0005   context run raw, frame 55, prompt v2: descartado (fora das branches)
│       ├── run-0006   context run raw, frame 55, prompt v3
│       ├── run-0007   context run raw, frame 61, prompt v3
│       ├── run-0008   bloqueado: context run voxel antes de o CLI aceitar --select
│       ├── run-0009   context run voxel, frame 55 (percepção do run-0006, reusada)
│       ├── run-0010   context run voxel, frame 61 (percepção do run-0007, reusada)
│       ├── run-0011   build raw bloqueado (semantic_scoring ligado pela camada base)
│       ├── run-0012   build raw  -> workspace/frame-10_31-raw/000001
│       ├── run-0013   build voxel -> workspace/frame-10_31-voxel/000001
│       └── branches/frame-10_31-{raw,voxel}
├── frame-10_31-raw/000001           # ContextMap 91a94edf924d74485a33587a805e02fb
└── frame-10_31-voxel/000001         # ContextMap 71613c69963db1f61eb8a1e2ecff0a90
```

Os registros das branches das tentativas v1 e v2, e as pastas de mapa vazias que elas reservavam,
foram movidos para fora do workspace (scratchpad da sessão), não apagados.

## Configuração

Pilha: `examples/v0.2.0/canonical-config.toml` + `base.template.toml` (canonical-2) +
`providers.toml` + `corridor-02-image-masks.toml` + `region-label.toml` + [`run.toml`](run.toml)
(+ [`voxel.toml`](voxel.toml) na fundação voxel), com `prompt_policy.scene=null` e, onde nenhum
modelo roda, `resources.providers={}`. Os comandos exatos estão em [`commands.sh`](commands.sh).

| Estágio | Esta run |
|---|---|
| Ingestão | janela 628–637 s (E0), `constant_offset` 78621358412432135 ns, calibração refinada (C2) |
| Estado | `external_pose` com o GT |
| Geometria | deskew `pose-interpolation-deskew-v1`; voxel: `scan-voxel-centroid-v1`, 0,10 m |
| Percepção | SAM2 tiny IoU 0,5 / estabilidade 0,80; sem cena; Qwen3-VL-4B 4-bit, `region-interaction/v3`, view `mask_outline_crop` (magenta, 2 px) |
| Fusão | baseline, canais `semantic_claims` + `visual_features` (sem scoring) |
| Resolução | geometria (0,5/0,5/0,9/0,5/0,3), crença C9 `resolved-belief-min-support-v1`, mínimo 2 |
| Relações | `next_to`, `touching`, `within_interaction_range`; 0,6 / 2,0 m; alcance de interação **derivado** do maior raio |

## O que mudou no código para a run

Tudo com teste antes (Red confirmado), local:

1. **Prompts `region-interaction/v2` e `v3`** (`visual_perception/semantic_prompt.py`). Ver
   "Prompt do fator de interação" abaixo.
2. **`context run --select visual_perception=<id> --catalog`**. Antes o `context run` aceitava
   `--select` e o ignorava, e executava a percepção de novo. Agora a percepção selecionada entra
   como fornecida, só a associação roda, e a `ContextRun` a cita como `reused`. Antes de
   executar, `check_context_selection` lê o manifest do run selecionado e recusa um que não foi
   construído sobre a sequência da fundação ou que não processou exatamente a seleção que a
   `ContextRun` grava. `publish_context_run` repete a verificação. Só `visual_perception` pode
   ser selecionado. A leitura fica em `runtime/perception_selection.py`, com exceção
   documentada no teste de fronteiras.
3. **`merge_calibration`** preserva os refinamentos da calibração fornecida (antes, a sequência
   publicada perdia o registro do refinamento).
4. `run.toml` desliga `semantic_scoring`: a fusão desta run não consome scores, e o estágio
   toma um run de percepção só, não os dois do build.

## Prompt do fator de interação (medido no frame 55)

| Prompt | Run | Regiões | Falhas de parse | Claims primárias com fator | Fatores |
|---|---|---|---|---|---|
| `region-interaction/v1` | run-0004 | 61 | 33 (54%): 31 com `interaction_factor` fora de `attributes`, 2 sem campo obrigatório | 0 de 28 | — |
| `region-interaction/v2` | run-0005 | 61 | 59 (97%): o claim do exemplo devolvido como a resposta inteira, sem `abstained`/`claims` | 2 de 2 | 0,8 ×2 |
| `region-interaction/v3` | run-0006 | 61 | **0** (6 abstenções) | 55 de 55 | 0,8 ×53, 0,5 ×2 |
| `region-interaction/v3` | run-0007 (frame 61) | 68 | **0** (5 abstenções) | 63 de 63 | 0,8 ×60, 0,5 ×3 |

A v2 foi um erro meu de desenho: o exemplo mostrava só o claim, e o modelo o copiou como a
resposta toda. A v3 mostra a resposta inteira, e um teste garante que o exemplo, preenchido,
passa no parser estrito. Resumos em `semantic-summary-*.json`.

## Resultados (medidos, `report.json`)

| | raw | voxel |
|---|---|---|
| Pontos do mapa | 2.264.716 | 393.054 |
| Percepção (mesmos runs nos dois) | run-0006, run-0007 | idem |
| Pontos visíveis, frame 55 / 61 | 38.189 / 37.632 | 6.494 / 6.665 |
| Pontos de suporte (soma), frame 55 / 61 | 38.353 / 33.096 | 6.124 / 5.554 |
| Observações sem suporte, 55 / 61 | 8 / 7 | 9 / 8 |
| Suportes de fusão | 68 | 57 |
| Entidades (semantic mapping) | 68 | 57 |
| Entidades resolvidas (= entidades do mapa) | 63 | 51 |
| Resolvidas vistas nos dois frames | 11 | 11 |
| Crença C9 | 54 tentative, 4 supported, 1 contested, 4 insufficient_evidence | 43 tentative, 5 supported, 3 insufficient_evidence |
| Resolvidas sem raio (sem primária) | 14 | 13 |
| Resolvidas com mais de um raio distinto | 2 | 3 |
| Raio de interação (mediana / p90 / máx) | 0,22 / 0,66 / 3,79 m | 0,29 / 0,95 / 3,78 m |
| Alcance derivado | 3,786 m | 3,784 m |
| Candidatos `next_to` / `touching` / `within_interaction_range` | 471 / 471 / 3.498 | 327 / 327 / 2.218 |
| Relações (todas as decisões) | 5.382 | 3.526 |
| `within_interaction_range` supported / rejected / unresolved | 236 / 1.386 / 1.876 | 192 / 804 / 1.222 |
| `next_to` supported | 360 | 292 |
| `touching` supported | 116 | 82 |
| `validate --full` | íntegro, 0 problemas, 3 avisos `dependency.optional_missing` | idem |

Crença `supported` (duas observações físicas da mesma label):
- raw: frame, door, shelf, bar; `contested`: a entidade wall × window (2 × 2);
- voxel: bar, shelf, frame, door e wall (a mesma entidade, aqui 2 × 1).

## Achados

Medidos:

- **O fator de interação quase não varia.** Com a v3, 113 de 118 fatores são 0,8 e 5 são 0,5,
  qualquer que seja o objeto (parede, porta, botão, faca). O raio fica ≈ 0,4 × o maior lado
  medido, e a relação `within_interaction_range` mede essencialmente o tamanho do objeto, não uma
  noção semântica de alcance. Na v1 e na v2 os fatores que vieram também eram quase todos 0,8.
- **O alcance derivado é definido por uma entidade de 10 m.** O maior raio (3,79 m) é o de uma
  entidade resolvida com hipóteses {chair, door, screw} e extensão 5,1 × 10,4 × 0,8 m, a mesma
  nos dois braços. Com esse alcance, 3.498 dos 4.440 candidatos do braço raw são
  `within_interaction_range`.
- **O contorno da view vaza para a semântica.** Na v3, 11 de 55 (frame 55) e 20 de 63 (frame
  61) claims primárias têm `attributes.color` rosa/magenta, a cor do contorno. Os rótulos desses
  claims são sobretudo "edge", "line", "frame" e, uma vez, "outline". Na v1, 10 de 28 descrições
  citavam a borda rosa. A contagem é por palavra-chave (heurística), não por anotação.
- **`region_kind` é "thing" em todos os claims**, inclusive parede e teto, também na v3, em que
  o exemplo não traz o valor.
- **Crença C9:** com dois frames a ~0,25 s, 4–5 entidades chegam a `supported`. As duas vistas
  são quase iguais, então esse suporte é fraco (a pergunta D7 segue aberta).

Hipóteses (não medidas):

- A entidade de 10 m pode vir de associação que vaza para fundo/paredes ou de uma fusão/resolução
  que junta regiões distantes; não foi investigado qual.
- Que o fator constante venha do modelo (4B, 4-bit) ou do enunciado não está separado; uma
  comparação com outro modelo ou com uma escala discreta responderia.
- Os rótulos "edge"/"line"/"frame" rosa sugerem que o modelo às vezes nomeia o próprio contorno;
  a view `tight_crop` sem contorno é a comparação natural (ablação C6).

## Identidades de código

- Fundações (run-0001, run-0003): `a3a175f+local.51d175b60ae9` (hash só de `git diff`).
- Context runs raw v3 (run-0006, run-0007): `a3a175f+local.63077e630372`.
- Context runs voxel, builds e mapas: `a3a175f+local.04d61c01a05c` (a mudança do `--select`
  entrou depois das context runs raw; ela não altera percepção nem associação).

A partir da segunda, o hash cobre `git diff HEAD` e os fontes não rastreados de `src/`
(fórmula em `commands.sh`).

## Viewer

`~/Projects/contextmap2-viewer/.env.run-frame-10_31` aponta os dois mapas e o run store:

```bash
cd ~/Projects/contextmap2-viewer
HOST=0.0.0.0 PORT=3000 npx nuxt preview --dotenv .env.run-frame-10_31
```

Verificado pela API numa instância temporária: os dois mapas aparecem, com mapa, trajetória e
semântica respondendo 200. A inspeção visual (modos Geometria e Labels, inspetor) não foi feita.

## Arquivos

- `run.toml`, `voxel.toml`, `timestamp-policy.json`, `code-identity.txt`, `commands.sh`
- `catalog-*.json`: catálogos de seleção (fundação, fundação voxel, percepção por frame)
- `ingest-*.json`, `branch-*.json`, `context-run-*.json`, `context-build-*.json`,
  `validate-*.json`: saídas JSON do CLI; `*.log`: stderr
- `semantic-summary-*.json`: parse e fator por prompt; `report.json`: métricas por braço
- `scripts/semantic_summary.py`, `scripts/run_report.py`, `scripts/perception_catalog.py`
