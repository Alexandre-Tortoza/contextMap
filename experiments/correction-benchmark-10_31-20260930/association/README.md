# C5 — Taxonomia de conflitos da associação 2D→3D e ablação offline (preliminar)

**Status:** preliminar. Usa as regiões do **E0** (SAM2 com limiares 0/0) e o mapa **sem deskew**.
Nenhuma política entra no código de produção nesta etapa. Tudo o que não foi medido está marcado
como hipótese.

Métricas: `contextmap.evaluation.association_conflicts`
(definições em `src/contextmap/evaluation/docs/association-conflicts.md`). Associação:
`experiments/semantic-geometry-investigation-20260930/scripts/assoc.py`, a reimplementação
validada bit a bit contra o artefato. A sanidade abaixo reconfirma isso.

## Como rodar

Da raiz do repositório, com o Python 3.12 do mise (numpy, scipy, PIL):

```bash
PY=python3.12  # o Python 3.12 do mise (numpy, scipy, PIL); o pacote não está instalado
PYTHONPATH=src $PY experiments/correction-benchmark-10_31-20260930/association/validate_association_conflicts.py
PYTHONPATH=src $PY experiments/correction-benchmark-10_31-20260930/association/c5_offline_ablation.py
```

- `e0_association.py`: entradas do E0. A pertença persistida vem da API pública
  (`SensorAssociationRunReader` + `geometry_index_of`), as máscaras de
  `../regions/e0_regions.py`, e a profundidade, o suporte e a oclusão por ponto de `assoc.py`
  (o artefato não persiste profundidade por ponto).
- Saídas: `results/e0_association_conflicts_validation.json` e
  `results/c5_offline_ablation.json`. Cada uma traz a proveniência: run de percepção
  `ebd2f3e3…`, run de associação `c744af1b…`, mapa `corridor-02--5659900d…`, identidade da
  calibração, HEAD `a3a175f`, hash do diff rastreado e sha256 dos dois módulos de métrica.
- Custo: a grade completa (30 braços) roda em 22 a 55 s de CPU, então não foi reduzida.

## 1. Validação contra a investigação (E0, calibração base)

| Métrica | p5 | módulo |
|---|---|---|
| pertença da reimplementação = artefato | — | sim (83/83 regiões) |
| visíveis / ocluídos | 23.948 / 1.516.151 | 23.948 / 1.516.151 |
| pontos associados / pertenças | 21.602 / 53.977 | 21.602 / 53.977 |
| histograma regiões/ponto | 1: 1.656, 2: 9.697, 3: 8.546, 4: 1.272, 5: 395, 6: 26, 7: 10 | idêntico |
| multi_mask_point_rate | 92,33 % | 92,33 % |
| máscaras com ≥ 30 pontos | 39 | 39 |
| contaminação por camada (ponderada) | 0,0042 | 0,0042 (0,00419); idêntica por máscara |
| ocluídos a ≤ 0,5 m do suporte | 12,187 % | 12,187 % |

Não há diferença a explicar. p5 grava a contaminação arredondada a 4 casas, e a comparação usa o
mesmo arredondamento.

## 2. Primeira taxonomia do E0

Parâmetros: aninhado = contenção 2D ≥ 0,9; duplicata = IoU ≥ 0,8 (valores do plano). Um ponto
recebe a classe mais severa entre os pares a que pertence: partial > duplicate > nested.

| Classe | Pontos multi-máscara / associados | Pares |
|---|---|---|
| partial | 12.489 (57,8 %) | 169 |
| nested | 7.457 (34,5 %) | 35 |
| duplicate | 0 | 0 |

- 100 % dos pontos multi-máscara ficam classificados, porque `partial` é a classe residual.
- Não há duplicatas porque a normalização já funde pares com IoU ≥ 0,8.
- A taxonomia depende muito do limiar de aninhamento. Com 0,8, partial fica em 55,3 % e nested
  em 37,0 %. Com 0,5, partial cai para 19,8 % e nested sobe para 72,6 %.
- Composição da classe partial (diagnóstico descritivo, em `partial_class_breakdown`):
  - 66 % das incidências (ponto, par) têm contenção 2D entre 0,5 e 0,9;
  - 82 % envolvem uma das 6 máscaras com área ≥ 10 % da área útil (0005, 0008, 0010, 0015,
    0017, 0022; por exemplo, 0005 "flooded hallway" e 0017, que cobre porta e parede).

## 3. Ablação offline (uma variável por eixo, grade completa)

Eixos:

- **Calibração:** base, ou corrigida com yaw +4,0°, pitch −1,75° e roll 0. É a mesma convenção de
  p4/p5 (`rotation_correction` = Rz(roll)·Rx(pitch)·Ry(yaw), aplicada no frame óptico).
- **Candidatos:**
  - mapa inteiro;
  - scans com início a ±50/±150/±500 ms da imagem;
  - scan em andamento no instante da imagem (substituto declarado do ±50 ms, abaixo);
  - alcance ≤ 20 m do centro óptico.
- **Oclusão:** `neighborhood_radius_cells` r ∈ {0, 1, 2}, com as margens atuais (célula 4 px,
  margem max(0,1 m; 2 %)).
- **Fixos:** regiões do E0, profundidade = alcance do raio, métricas com os parâmetros de p5.

Desvios declarados:

- **±50 ms pelo início do scan é vazio.** Os inícios mais próximos estão a −50,26 ms (scan 22) e
  a +50,60 ms (scan 23). Nos dois braços, esse candidato fica registrado em `skipped_arms`.
  Para cobrir a intenção (o scan sincronizado, como no V1), acrescentei o braço "scan em
  andamento": o último scan com início ≤ t_imagem, que é o scan 22. A varredura dele dura
  100,84 ms e contém o instante da imagem.
- Os braços de scan usam as coordenadas do mapa, **sem deskew**. Por isso não são comparáveis
  ao "scan local com deskew" de p5, que teve contaminação de 2,3 % com a oclusão do pipeline.
- Sanidade: base|mapa|r2 = artefato do E0 (23.948 visíveis, 21.602 associados, mesmos pontos), e
  corrigida|mapa|r2 = `fit_p4|pipeline` de p5 (23.208 / 20.857, mesmo histograma, contaminação
  0,0018).

Colunas: "E0 mantido" é a fração dos 21.602 pontos associados do E0 que continuam associados.
"camada" é a contaminação ponderada, com o número de máscaras medidas entre parênteses. "ocl.
≤ 0,5 m" é a fração de ocluídos a ≤ 0,5 m do suporte.

| calib. | candidatos | r | candidatos (pts) | projetados | na imagem | suporte válido | visíveis | associados | E0 mantido % | regiões c/ pts | multi % | nested % | partial % | camada % (máscaras) | ocl. ≤ 0,5 m % (n / ocluídos) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | mapa inteiro | 0 | 2.322.324 | 1.783.217 | 1.549.066 | 1.540.099 | 134.471 | 116.944 | 100,0 | 83 | 79,5 | 28,9 | 50,7 | 4,05 (74) | 21,8 (306.635 / 1.405.628) |
| base | mapa inteiro | 1 | 2.322.324 | 1.783.217 | 1.549.066 | 1.540.099 | 41.702 | 37.085 | 100,0 | 81 | 88,5 | 34,9 | 53,7 | 1,33 (52) | 15,1 (226.615 / 1.498.397) |
| base | mapa inteiro | 2 | 2.322.324 | 1.783.217 | 1.549.066 | 1.540.099 | 23.948 | 21.602 | 100,0 | 75 | 92,3 | 34,5 | 57,8 | 0,42 (39) | 12,2 (184.768 / 1.516.151) |
| base | início ±150 ms | 0 | 57.515 | 40.525 | 18.012 | 17.750 | 15.750 | 12.910 | 14,5 | 60 | 84,5 | 29,2 | 55,3 | 10,18 (51) | 75,8 (1.517 / 2.000) |
| base | início ±150 ms | 1 | 57.515 | 40.525 | 18.012 | 17.750 | 11.318 | 9.553 | 14,5 | 59 | 85,0 | 29,2 | 55,8 | 4,78 (39) | 78,2 (5.031 / 6.432) |
| base | início ±150 ms | 2 | 57.515 | 40.525 | 18.012 | 17.750 | 7.455 | 6.718 | 14,5 | 47 | 87,9 | 30,3 | 57,6 | 2,76 (29) | 65,6 (6.754 / 10.295) |
| base | início ±500 ms | 0 | 287.046 | 200.693 | 98.198 | 96.703 | 59.960 | 50.528 | 42,6 | 63 | 87,1 | 34,1 | 53,0 | 5,47 (56) | 71,6 (26.313 / 36.743) |
| base | início ±500 ms | 1 | 287.046 | 200.693 | 98.198 | 96.703 | 32.873 | 28.753 | 42,6 | 58 | 89,5 | 33,2 | 56,3 | 1,50 (42) | 66,5 (42.421 / 63.830) |
| base | início ±500 ms | 2 | 287.046 | 200.693 | 98.198 | 96.703 | 22.336 | 19.502 | 42,6 | 54 | 91,8 | 30,9 | 60,9 | 0,47 (30) | 62,0 (46.073 / 74.367) |
| base | scan em andamento | 0 | 28.709 | 19.793 | 8.502 | 8.358 | 7.649 | 6.115 | 7,4 | 59 | 83,6 | 29,1 | 54,5 | 14,65 (41) | 82,9 (588 / 709) |
| base | scan em andamento | 1 | 28.709 | 19.793 | 8.502 | 8.358 | 6.277 | 4.771 | 7,4 | 58 | 84,1 | 28,5 | 55,6 | 6,17 (34) | 76,6 (1.595 / 2.081) |
| base | scan em andamento | 2 | 28.709 | 19.793 | 8.502 | 8.358 | 4.637 | 3.493 | 7,4 | 51 | 86,2 | 30,1 | 56,1 | 4,54 (28) | 62,9 (2.339 / 3.721) |
| base | alcance ≤ 20 m | 0 | 2.307.030 | 1.772.187 | 1.538.377 | 1.529.410 | 134.471 | 116.944 | 100,0 | 83 | 79,5 | 28,9 | 50,7 | 4,05 (74) | 22,0 (306.635 / 1.394.939) |
| base | alcance ≤ 20 m | 1 | 2.307.030 | 1.772.187 | 1.538.377 | 1.529.410 | 41.702 | 37.085 | 100,0 | 81 | 88,5 | 34,9 | 53,7 | 1,33 (52) | 15,2 (226.615 / 1.487.708) |
| base | alcance ≤ 20 m | 2 | 2.307.030 | 1.772.187 | 1.538.377 | 1.529.410 | 23.948 | 21.602 | 100,0 | 75 | 92,3 | 34,5 | 57,8 | 0,42 (39) | 12,3 (184.768 / 1.505.462) |
| corrigida | mapa inteiro | 0 | 2.322.324 | 1.774.549 | 1.550.499 | 1.542.117 | 130.330 | 113.466 | 93,0 | 83 | 72,0 | 31,7 | 40,3 | 0,37 (72) | 21,9 (309.385 / 1.411.787) |
| corrigida | mapa inteiro | 1 | 2.322.324 | 1.774.549 | 1.550.499 | 1.542.117 | 41.033 | 36.200 | 93,0 | 82 | 86,0 | 35,5 | 50,4 | 0,33 (52) | 14,9 (224.175 / 1.501.084) |
| corrigida | mapa inteiro | 2 | 2.322.324 | 1.774.549 | 1.550.499 | 1.542.117 | 23.208 | 20.857 | 86,6 | 71 | 90,7 | 32,5 | 58,3 | 0,18 (39) | 12,0 (182.070 / 1.518.909) |
| corrigida | início ±150 ms | 0 | 57.515 | 40.456 | 18.047 | 17.816 | 15.821 | 12.538 | 13,6 | 58 | 85,2 | 28,1 | 57,2 | 8,32 (46) | 77,4 (1.544 / 1.995) |
| corrigida | início ±150 ms | 1 | 57.515 | 40.456 | 18.047 | 17.816 | 11.395 | 9.370 | 13,6 | 56 | 84,8 | 27,7 | 57,1 | 3,84 (39) | 78,5 (5.039 / 6.421) |
| corrigida | início ±150 ms | 2 | 57.515 | 40.456 | 18.047 | 17.816 | 7.453 | 6.618 | 13,6 | 51 | 86,7 | 28,0 | 58,7 | 1,25 (29) | 66,0 (6.843 / 10.363) |
| corrigida | início ±500 ms | 0 | 287.046 | 199.379 | 98.808 | 97.201 | 58.777 | 47.961 | 39,3 | 63 | 86,6 | 33,7 | 52,9 | 2,25 (56) | 70,5 (27.107 / 38.424) |
| corrigida | início ±500 ms | 1 | 287.046 | 199.379 | 98.808 | 97.201 | 31.109 | 26.591 | 39,3 | 60 | 87,7 | 34,2 | 53,5 | 1,12 (40) | 65,7 (43.444 / 66.092) |
| corrigida | início ±500 ms | 2 | 287.046 | 199.379 | 98.808 | 97.201 | 20.490 | 17.445 | 38,7 | 56 | 89,9 | 33,2 | 56,7 | 0,31 (29) | 61,6 (47.260 / 76.711) |
| corrigida | scan em andamento | 0 | 28.709 | 19.752 | 8.537 | 8.420 | 7.722 | 5.927 | 7,0 | 58 | 84,4 | 27,5 | 56,9 | 9,16 (37) | 85,1 (594 / 698) |
| corrigida | scan em andamento | 1 | 28.709 | 19.752 | 8.537 | 8.420 | 6.317 | 4.599 | 7,0 | 57 | 84,1 | 27,1 | 56,9 | 5,43 (31) | 76,0 (1.599 / 2.103) |
| corrigida | scan em andamento | 2 | 28.709 | 19.752 | 8.537 | 8.420 | 4.709 | 3.392 | 7,0 | 50 | 86,4 | 28,0 | 58,3 | 1,44 (22) | 61,8 (2.295 / 3.711) |
| corrigida | alcance ≤ 20 m | 0 | 2.307.030 | 1.763.519 | 1.539.810 | 1.531.428 | 130.330 | 113.466 | 93,0 | 83 | 72,0 | 31,7 | 40,3 | 0,37 (72) | 22,1 (309.385 / 1.401.098) |
| corrigida | alcance ≤ 20 m | 1 | 2.307.030 | 1.763.519 | 1.539.810 | 1.531.428 | 41.033 | 36.200 | 93,0 | 82 | 86,0 | 35,5 | 50,4 | 0,33 (52) | 15,0 (224.175 / 1.490.395) |
| corrigida | alcance ≤ 20 m | 2 | 2.307.030 | 1.763.519 | 1.539.810 | 1.531.428 | 23.208 | 20.857 | 86,6 | 71 | 90,7 | 32,5 | 58,3 | 0,18 (39) | 12,1 (182.070 / 1.508.220) |

### O que foi medido

1. **Alcance ≤ 20 m não muda a associação neste frame.** Visíveis, associados e todas as métricas
   de conflito são idênticos aos do mapa inteiro nos 6 pares (calibração × r). O alcance só remove
   15.294 candidatos (e ocluídos) além de 20 m.
2. **O raio de oclusão domina o tamanho da população.** De r2 para r0 (mapa inteiro), os
   associados sobem 5,4× (21.602 → 116.944 na base; 20.857 → 113.466 na corrigida).
   - A contaminação por camada sobe de 0,42 % para 4,05 % com a calibração base, mas só de
     0,18 % para 0,37 % com a corrigida.
   - A fração de ocluídos a ≤ 0,5 m do suporte vai de 12 % (r2) para 22 % (r0).
3. **Janelas temporais pelo início do scan reduzem muito a população e pioram a contaminação**,
   sempre em relação ao mapa inteiro com o mesmo r e a mesma calibração:
   - só 7 a 43 % dos associados do E0 continuam;
   - de 47 a 63 regiões recebem pontos, contra 71 a 83 no mapa inteiro;
   - a contaminação por camada é maior nos 18 braços de scan (por exemplo, r0 corrigida:
     0,37 % no mapa inteiro, 2,25 a 9,16 % nos scans);
   - 62 a 85 % das oclusões ficam a ≤ 0,5 m do suporte, em números absolutos pequenos
     (698 a 76.711 ocluídos).
4. **O multi-pertencimento continua alto em todos os 30 braços** (72,0 a 92,3 %). Nenhum eixo o
   reduz abaixo de 72 %. Faixas: duplicate sempre 0; partial de 40 a 61 %; nested de 27 a 36 %.
   O mínimo é corrigida|mapa|r0.
5. **A calibração corrigida reduz o partial em r0** (50,7 → 40,3 %). Em r2 não reduz
   (57,8 → 58,3 %).

### Hipóteses (não isoladas)

- **H-C5a** (o multi-pertencimento residual é parte-todo) **não é decidível com as regiões do
  E0.** O multi-pertencimento vem sobretudo do conjunto de máscaras, não dos candidatos nem da
  oclusão: é alto em todos os braços, e o partial é dominado por máscaras grandes de "cena" e
  por pares com contenção de 0,5 a 0,9, típicos de máscaras fragmentadas ou salpicadas.
  Hipótese: os limiares do C4 reduzem o partial. Teste: repetir esta grade com as regiões do E3.
- **H-C5b / H7** (a janela de 20×20 px descarta a mesma superfície). Hipótese compatível com o
  medido, mas não provada: r0 multiplica por 5,4 os associados e, **só com a calibração
  corrigida**, mantém a contaminação por camada ≤ 0,37 %. A métrica de camada não vê um objeto
  errado à mesma profundidade nem os fantasmas do acúmulo sem deskew. Antes de qualquer política
  de raio, é preciso medir o recall e a precisão de pontos com a anotação fraca do C0, depois do
  C2 e do C3. Com a calibração base, r0 multiplica a contaminação por ~10: a calibração vem antes.
- **Janela temporal:** hipótese de que ela só é útil com deskew (C3) e com uma regra de oclusão
  própria para geometria esparsa. Num scan isolado, a vizinhança de 5×5 células oclui
  principalmente pontos a ≤ 0,5 m do suporte, provavelmente a mesma superfície. Não é
  candidata agora.
- **Alcance 20 m:** sem efeito neste frame. É coerente com manter `association_max_range_m`
  desligado no config canônico.

### Não medido / pendências

- **Recall de pontos dos objetos anotados** (critério de aceite do C5: não perder mais de 5 %):
  não há anotação do frame 55 (`annotations/` vazio; bloqueio humano do C0).
- **E1c/E3:** calibração versionada (C2) + deskew (C3) + regiões do C4. Esta grade é sobre
  E0 + correção offline.
- A oclusão "z-buffer por pixel exato" de p5 não entrou na grade (não foi pedida).
