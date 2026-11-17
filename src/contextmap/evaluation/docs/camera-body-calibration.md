# Avaliação da calibração câmera↔corpo

Este documento descreve `src/contextmap/evaluation/camera_body_calibration.py`: medidas objetivas para comparar uma calibração câmera↔corpo com outra, em especial a base de um dataset e a mesma base com um `CalibrationRefinement` (ver `src/contextmap/ingestion/docs/calibration.md`).

Nenhuma função escolhe correção nem altera artefato. O experimento fornece as entradas e lê os números; a grade de busca vive no script do experimento que gera o documento de refinamento.

## Medidas

| função | o que mede | unidade | melhor |
|---|---|---|---|
| `photometric_inconsistency(groups, projection=, correction=)` | variância da textura local que o mesmo ponto do mapa amostra em vários frames | textura² (adimensional) | menor |
| `pixel_displacement(first, second)` | distância entre duas projeções dos mesmos pontos: mediana, p10, p90 | px da imagem bruta | — |
| `reprojection_residual(projection=, points_camera_m=, observed_pixels=)` | distância entre o pixel anotado e a projeção do ponto 3D de cada correspondência: mediana, p90 | px da imagem bruta | menor |
| `relative_improvement(baseline, candidate)` | `(baseline − candidate) / baseline` de uma medida em que menor é melhor | fração | maior |

### Inconsistência fotométrica

É a definição do estimador da investigação (`experiments/semantic-geometry-investigation-20260930/scripts/p4_calibration_photometric.py`), que replica o critério do V1 (#285), fixada aqui para que a estimativa e a validação meçam a mesma coisa:

1. um `PhotometricGroup` traz, para `N` pontos e `F` frames de um trecho curto, as coordenadas de cada ponto no frame óptico **base** de cada frame (`(N, F, 3)`, metros), a visibilidade (`(N, F)`) e a textura local normalizada de cada frame (`(H, W)`);
2. a visibilidade é decidida uma vez, com a calibração base, por quem monta o grupo; a correção só muda onde cada ponto visível lê a textura, então toda correção candidata é julgada sobre as mesmas observações;
3. cada ponto visível é girado (`p = Rcorr·p_base`), projetado pelo `CameraProjection` de `sensor_association`, arredondado para o pixel mais próximo (`np.round`) e amostrado quando o pixel existe na textura;
4. um ponto com pelo menos `MIN_SAMPLES_PER_POINT` (3) amostras contribui a variância populacional delas;
5. os pontos são somados entre grupos: o resultado é a soma das variâncias sobre o total de pontos, não a média das médias dos grupos.

Nenhum ponto com amostras suficientes é erro (`ValueError`), não zero.

Diferença medida em relação ao script da investigação: a fórmula MEI independente de `p4` só exigia `z + xi·r > 0`. Com `xi = 1,563 > 1`, ela dobra para dentro da imagem pontos atrás da câmera. No corridor-02, 4,7 % dos pares (ponto, frame) que o `p4` considerou visíveis estavam a mais de 164° do eixo óptico. A projeção do pipeline recusa esses pontos (horizonte da esfera em `z > −min(xi, 1/xi)·alcance`). Por isso os números desta medida não coincidem com os de `results/p4_calibration_photometric.json`, e o experimento do C2 registra os dois braços.

A textura (passa-alta normalizada por energia local) e a seleção dos pontos (deskew, alcance, oclusão por z-buffer) dependem de SciPy e ficam no experimento; esta medida recebe os arrays prontos.

### Resíduo de reprojeção

As correspondências 2D↔3D precisam de anotação humana (piloto fraco do C0). Um ponto 3D que o modelo de câmera não projeta é contado em `unprojectable_count` e fica fora da estatística, nunca silenciosamente. Até existir anotação, essa medida não tem número no C2.

## Relação com o C2

`experiments/correction-benchmark-10_31-20260930/calibration/estimate_refinement.py` reexecuta a grade da investigação com `photometric_inconsistency` como custo, compara cada número com `results/p4_calibration_photometric.json` e escreve o documento `calibration-refinement/1` com as métricas de validação (ajuste/validação, melhoria relativa, deslocamento em pixel no frame 55).
