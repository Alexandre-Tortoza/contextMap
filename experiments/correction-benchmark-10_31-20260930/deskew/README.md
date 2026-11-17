# C3 — validação do deskew in-repo (`pose-interpolation-deskew-v1`)

Pergunta: o deskew implementado em `src/contextmap/geometric_mapping/deskew.py`, aplicado pelo
caminho real de montagem (`assemble_geometry_inputs_from_artifacts(..., deskew=...)`), reproduz o
deskew offline da investigação (`semantic-geometry-investigation-20260930`, parte 5) na janela
628–637 s do corridor-02?

## Como reproduzir

```bash
PYTHONPATH=src python \
    experiments/correction-benchmark-10_31-20260930/deskew/validate_deskew.py \
    --work-dir /tmp/deskew-work   # os 2 mapas (~290 MB) vão para fora do repositório
```

Saída: `results/deskew_validation.json`. O script roda em ~6 min (wall clock, 1 máquina) e usa scipy apenas na
métrica (reaproveitada de `p3_map_consistency.py`).

## Entradas (somente leitura)

- `workspace-old/frame-10_31-raw/sequences/corridor-02/aa0551d9…` (SequenceArtifact, 89 scans);
- `workspace-old/frame-10_31-raw/state_estimation` (GT externa, ~10 Hz);
- a política de lookup (`interpolated`, gap máx. 400 ms) e a de correção (`raw=accept`,
  `unknown=warn`) lidas do `config.json` do mapa E0, não redefinidas aqui.

Campo de tempo por ponto medido nos 89 scans: `time`, float32, offset 18 do registro de 22
bytes, em **segundos**, relativo ao header (mín. 0–2,3 µs, máx. 100,78–100,84 ms, monotônico
não decrescente em todos os scans). O período entre headers é 100,80–100,86 ms, então o header
é o **início** do scan. Política: `time_field="time"`, `SECONDS`, `SCAN_TIMESTAMP`, bin de 1 ms.

## Resultados (medidos)

Métrica da investigação: ponto-plano contra o scan de referência `velodyne_points-000022`, com
51 scans a ≤ 3 s, pontos a < 12 m (filtro nas coordenadas brutas), planaridade < 0,02 e raio de
0,25 m.

| braço | mediana | p90 | p95 | correspondências |
| --- | --- | --- | --- | --- |
| in-repo sem deskew | 3,282 cm | 10,048 cm | 12,668 cm | 1.080.456 |
| **in-repo com deskew** | **2,155 cm** | **6,841 cm** | **10,056 cm** | 1.087.707 |
| deskew offline da investigação (recalculado) | 2,161 cm | 6,852 cm | 10,089 cm | 1.089.956 |

- Diferença in-repo × offline: mediana 0,28 %, p90 0,16 % (alvo ≤ 5 %). O braço offline
  recalculado bate com os valores gravados pela investigação (2,161 / 6,852 cm).
- Alvos do plano: mediana ≤ 2,4 cm ✔, p90 ≤ 7,2 cm ✔.
- **Sem deskew, o caminho in-repo reproduz o `geometry.bin` do E0 byte a byte**
  (`sha256:d4e4b03e…`, 81 scans, 2.322.324 pontos).
- Com deskew: `sha256:59b225db…`, **79 scans**, 2.264.716 pontos, todos `corrected`.
- Concordância ponto a ponto in-repo × offline (mesmos pontos, coordenadas no mapa): mediana
  2,8 mm, p99 9,1 mm, máx. 2,9 cm a < 12 m; máx. 7,8 cm a ≥ 12 m. Os dois diferem no bin
  (1 ms no múltiplo mais próximo, limitado aos pontos, contra 5 ms no centro do bin). A
  mediana corresponde ao erro de tempo médio do bin de 5 ms (~1,25 ms × 2,1 m/s ≈ 2,6 mm) e o
  erro cresce com o alcance, como faria um erro angular. **Hipótese**, não isolada.
- `ScanOverlapReport` in-repo, com todos os pares adjacentes do mapa e 1000 pontos por scan
  (protocolo no JSON):

| braço | pares | mediana | p90 | p95 | fração > 5 cm |
| --- | --- | --- | --- | --- | --- |
| sem deskew | 80 | 1,038 cm | 4,122 cm | 5,531 cm | 6,48 % |
| com deskew | 78 | 0,963 cm | 3,788 cm | 4,955 cm | 4,88 % |

  Os conjuntos de pares diferem em 2 scans (ver abaixo), então a comparação não é pareada.
- Custo: a montagem com deskew levou 0,69 s contra 0,39 s sem deskew (89 scans, leitura da
  sequência incluída).

## Scans a mais recusados (critério "nenhum scan a mais rejeitado": **não atendido**)

`velodyne_points-000054` e `-000075` são recusados com `deskew_rejected`. A pose do header de
cada um é aceita, mas o próximo bin (+1 ms) já está no intervalo de GT de 605 ms (54) ou de
403 ms (75), acima da tolerância de 400 ms. A GT tem uma amostra quase coincidente com cada
header (delta mediano de 80 ns), então quase o scan inteiro cai no gap. Sem deskew, esses dois
scans entram no mapa com todos os pontos colocados pela pose do início. É a mesma regra que já
recusa os scans 55–59 e 76–78. Aceitá-los exigiria outra tolerância de lookup, uma decisão
separada que não foi tomada aqui.

A primeira execução recusava também os scans 53 e 74. O último bin (τ ≈ 100,84 ms) arredondava
para 101 ms, depois da amostra de GT seguinte, dentro do gap, onde nenhum ponto foi medido. A
correção limita o instante de cada bin aos instantes dos seus pontos (teste de regressão
`test_a_time_bin_is_never_looked_up_after_the_last_instant_it_serves`).

## Limites

- Um único trecho (628–637 s) e uma única referência de GT (~10 Hz, gaps de 101–605 ms). O
  modelo de velocidade constante entre amostras não foi comparado com IMU.
- A métrica é de consistência interna (ponto-plano entre scans), cega a deslizamentos ao longo
  da superfície. Não é acurácia absoluta.
