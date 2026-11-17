# C4 — Métricas estruturais de região: validação no E0

O que existe aqui:

- o módulo `contextmap.evaluation.region_structure`, com as definições em
  `src/contextmap/evaluation/docs/region-structure.md`;
- a validação desse módulo contra as medidas da investigação
  (`semantic-geometry-investigation-20260930/results/p6_mask_quality.json`).

A grade de limiares do SAM2 (E2/E3-g) **não** foi rodada aqui: ela exige re-executar a percepção.
Este diretório entrega a régua que ela vai usar.

## Como rodar

```bash
PY=python3.12  # o Python 3.12 do mise (numpy, scipy, PIL); o pacote não está instalado
PYTHONPATH=src $PY experiments/correction-benchmark-10_31-20260930/regions/validate_region_structure.py
```

- `e0_regions.py`: lê só pelas APIs públicas as 83 regiões aceitas de
  `workspace-old/frame-10_31-raw`, com `PerceptionRunReader` e `MaskStoreReader` (este verifica
  o hash de cada máscara). Lê também a contagem de propostas do backend (431, da auditoria de
  discovery) e a área útil (região válida menos o ego, dos mesmos `.npy` que a investigação usou).
- Saída: `results/e0_region_structure_validation.json`. Contém:
  - a comparação com p6 e a checagem por máscara;
  - o oráculo de componentes;
  - o report completo com IoU de duplicata 0,5 (o limiar de p6) e 0,8 (o do plano);
  - a proveniência: run `ebd2f3e3…`, digest do backend SAM2, sha256 dos `.npy`, HEAD, hash do
    diff rastreado e sha256 dos módulos de métrica.

## Resultado

Conectividade 4 (a mesma de p6: `scipy.ndimage.label` em 2D). Contenção ≥ 0,8.

| Métrica | p6 | módulo |
|---|---|---|
| máscaras | 83 | 83 (431 propostas → `accepted_mask_ratio` 0,193) |
| área p10 / mediana / p90 (px) | 270,6 / 1.100 / 7.892 | idêntica |
| fragmentadas (> 1 componente) | 68,7 % | 68,7 % (57/83) |
| máscaras com > 10 componentes / mediana de componentes | 20 / 3 | 20 / 3 |
| máscaras ≥ 80 % dentro de outra | 43/83 | 43/83 |
| pares com IoU ≥ 0,5 | 13 | 13 (com IoU ≥ 0,8: 0) |
| pares com sobreposição > 0 | 9,49 % dos pares | 323 pares (9,49 %) |
| pares contidos ≥ 0,8 | 57 (ordenados) | 56 (não ordenados) |
| pixels cobertos com 1 máscara | 15,1 % | 15,1 % (84,9 % com ≥ 2) |
| histograma de aninhamento | 25.455 / 66.005 / 60.180 / 11.240 / 4.041 / 1.021 / 226 | idêntico, máximo 7 |
| cobertura da área útil | 70,8 % | 70,8 % |

Por máscara, a área, os componentes e a maior fração dentro de outra são idênticos nas 83.

Diferenças, ambas de definição:

1. **Pares contidos.** p6 conta pares ordenados: 57 = 56 não ordenados + 1 par em que as duas
   direções passam de 0,8. O módulo conta pares não ordenados.
2. **"Contida em" das máscaras 0027, 0033 e 0053.** Elas não tocam nenhuma outra. p6 grava o
   argmax de um vetor de zeros (`region-0001`); o módulo grava `None`.

Oráculo: a contagem de componentes (só numpy) coincide com `scipy.ndimage.label` nas 83 máscaras
reais e em 400 máscaras aleatórias de 48×64, com densidade de 0,2 a 0,8.

## Pendências

- `known_object_recall` está testado, mas não foi medido no frame 55: não há anotação fraca
  (`../annotations/` vazio; bloqueio humano do C0). É ele que decide o critério de seleção do C4
  (D4).
- A grade E2/E3-g do C4 (re-execução do SAM2 por célula) fica para a etapa do C4. Cada arm deve
  chamar `measure_region_structure` com `proposal_count` e `usable_area` e declarar os limiares.
