# C2 — calibração câmera↔corpo refinada, versionada e reproduzível

Correção da extrínseca `T_epson_d` do corridor-02 como **dado versionado**: um documento
`calibration-refinement/1` aplicado uma vez na ingestão por
`contextmap.ingestion.apply_calibration_refinement`. O contrato está em
`src/contextmap/ingestion/docs/calibration.md`; as medidas, em
`src/contextmap/evaluation/docs/camera-body-calibration.md`.

## Arquivos

| arquivo | o que é |
|---|---|
| `corridor-02-camera-body-refinement-v1.json` | o refinamento: yaw +4,0°, pitch −1,75°, roll 0 sobre `T_epson_d` (digest de base `sha256:13220205…`), com método, identidade de código, digest do GT, sha256 dos YAMLs e evidência fit/held-out |
| `estimate_refinement.py` | estimador reprodutível por grade (dois braços, ver abaixo); gera o documento |
| `results/p4-reproduction.json` | braço que reproduz `results/p4_calibration_photometric.json` da investigação |
| `results/estimate.json` | braço que gera o documento: números, perfis, por grupo, deslocamento no frame 55 |
| `build_refined_calibration.py` | base do E0 (`ingest_window_fresh.build_calibration`) + refinamento → `CalibrationSet` do braço E1 |
| `refined-calibration.json` | o `calibration.json` que a sequência do E1 deve conter (saída do script acima) |

## Reproduzir

```bash
cd experiments/correction-benchmark-10_31-20260930/calibration
PY=~/.cache/contextmap2-audit/venv/bin/python   # numpy, scipy, PIL, contextmap editável
$PY estimate_refinement.py --arm p4-reproduction   # ~6 min em CPU
$PY estimate_refinement.py --arm estimate          # ~5 min em CPU; reescreve o documento
cd ../../.. && PYTHONPATH=src $PY experiments/correction-benchmark-10_31-20260930/calibration/build_refined_calibration.py
```

Entradas: `workspace-old/frame-10_31-raw` (sequência, trajetória GT persistida, mapa, máscaras do
frame 55), `examples/v0.2.0/assets/corridor-02/camera_1-{valid,ego}.npy` e
`datasets/corridor-02/*.yaml` + `corridor-02-gt.txt`. O deskew por ponto é o offline da
investigação (`p3_time_offset_deskew.place`), não o do repo (C3 ainda não existe).

## Resultados

Mesma seleção de p4: 7 grupos de 8 frames na janela 628–637 s, 60 000 pontos por grupo, grupos
pares = ajuste (4), ímpares = validação (3). Roll fixo em 0 (D3 continua aberta).

| braço | projeção | ótimo (ajuste) | ajuste base → refinado | validação base → refinado | melhoria na validação | V1 (4,25/−2,25) na validação |
|---|---|---|---|---|---|---|
| `p4-reproduction` | fórmula MEI de p4 (dobra) | 4,0 / −1,75 | 0,20405 → 0,13588 | 0,19177 → 0,13293 | −30,7 % | 0,13214 (−31,1 %) |
| `estimate` | projeção do pipeline | **4,0 / −1,75** | 0,20238 → 0,13036 | 0,18966 → 0,12907 | **−31,9 %** | 0,12790 (−32,6 %) |

- **Reprodução:** o braço `p4-reproduction` reproduz todos os números de
  `results/p4_calibration_photometric.json` com desvio máximo **0,0** (inconsistências, por grupo,
  perfis e ótimo). O harness (seleção, textura, grade) e a métrica do repo são, portanto, idênticos
  ao p4 quando recebem a mesma projeção.
- **Por que há dois braços (medido):** a fórmula MEI independente de p4 (`common.mei_project`) só
  exige `z + xi·r > 0`. Com `xi = 1,563 > 1`, ela dobra para dentro da imagem pontos atrás da
  câmera. Das 2 641 975 visibilidades (ponto, frame) que o p4 aceitou, **124 525 (4,7 %)** eram
  de pontos a mais de 164° do eixo óptico (2 045–4 785 pontos por grupo com ao menos uma
  visibilidade dobrada). Esses pontos também ocluíam pontos reais no z-buffer da seleção. A
  projeção do pipeline os recusa (horizonte em `z > −min(xi, 1/xi)·alcance`). O documento usa o
  braço `estimate`.
- **Efeito no número:** o ótimo não muda (4,0/−1,75 nos dois braços); a melhoria na validação
  passa de −30,7 % para −31,9 %. Por grupo (braço `estimate`), o ótimo varia de yaw 2,0–4,25° e
  pitch −0,5…−2,75°. Na definição de p4, a faixa era 3,0–4,5° e −0,25…−2,75°.
- **Deslocamento no frame 55** (pontos visíveis na base, mapa acumulado, oclusão do pipeline):
  melhor ajuste mediana **22,43 px** (p10 20,54; p90 23,33; 22 725 pontos); correção do V1
  24,37 px (p10 22,16; p90 25,39), igual a `p5_association_ablation.json`.
- **Resíduo de reprojeção:** **não medido**. Não existem ainda as ~10 correspondências 2D↔3D
  anotadas do C0 (`annotations/` vazio); a função `reprojection_residual` está pronta.

Aceite do plano: (a) "−31 % ± 3 pp nos mesmos 3 grupos de validação" → −31,9 % (e −30,7 % na
definição de p4). (b) ≥ 2 trechos disjuntos de outras partes do bag: **não feito**. (c) resíduo de
reprojeção: **não medido**. (d) contaminação das regiões finas: não medida aqui (depende do E1).

## Identidade

- base (E0): `calibration_identity = sha256:f7d3288e…`, `calibration_source_hash = sha256:4fe08c29…`;
  inalteradas pela mudança (fixadas em `tests/ingestion/test_calibration_refinement.py`);
- refinada (E1): `calibration_identity = sha256:2d115ca8cd8e8b2b1cc6f178907f8fd7c4da22eeeed11d74e5d803528a684c8e`.

A identidade cobre id, alvo, digest de base e correção. Método, código, pose, YAMLs e métricas
ficam fora. Reestimar com outra evidência e a mesma correção não muda a identidade.

## Decisões

- **Composição:** `R' = R_base·Rcorrᵀ`, `t' = t`. O plano escrevia `R_base·Rcorr`, o que está
  errado na convenção `T_parent_child`. A derivação foi conferida contra o p4 (`cams @ rot.T`),
  contra o V1 (`T_camera_lidar' = C·T_camera_lidar`, translação inclusive) e contra o pixel do
  teste D.
- **Convenção:** a mesma fórmula do V1 e do p4 (`Rz(roll)·Rx(pitch)·Ry(yaw)` aplicada ao frame
  óptico base). O texto é o do V1, com a nota de eixos em inglês. Yaw > 0 → `+u`; pitch > 0 → `−v`.
  Os dois sentidos são testados.
- **Schema do `CalibrationSet`:** fica em `0.1.0` nesta mudança. A chave `refinements` é opcional,
  omitida quando vazia, e o decode aceita sua ausência. O bump para `0.2.0` previsto no plano
  (i) exige a transição `CalibrationSet 0.1.0 → 0.2.0` no `CHANGELOG.md` [Não lançado], fora do
  escopo deste passo, porque `tests/packaging/test_release_scope.py` recusa a mudança sem ela;
  (ii) mudaria a identidade de **toda** calibração nova sem refinamento, porque
  `calibration_identity` cobre `schema_version`. Isso contraria o requisito de manter as
  identidades sem refinamento e desalinharia o E0 re-rodado. Fica para decisão.
- **Guarda extra:** o frame filho do alvo precisa ser o frame de uma entrada com `camera_model`.
  Sem câmera, yaw/pitch/roll ópticos não têm significado.
- **D3 (roll):** não decidida. O ajuste fixa roll = 0, como o p4; o V1 estimou roll −1,5° e −0,5°
  nas duas metades.

## Pendências (hipóteses ainda não validadas)

- H-C2: a correção generalizar para outros trechos do bag. Os 7 grupos vêm da mesma janela de 9 s,
  intercalados e correlacionados.
- Re-medir com o deskew do repo (C3) em vez do deskew offline da investigação.
- Resíduo de reprojeção com as correspondências anotadas do C0.
- Ingerir o braço E1 com `build_refined_calibration()`; `geometry.bin` deve ficar byte-idêntico
  ao E0, com `map_id` diferente.
