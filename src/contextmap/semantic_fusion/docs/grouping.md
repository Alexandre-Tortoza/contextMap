# Agrupamento por observação física

`group_by_physical_observation` agrupa correspondências regionais por `SourceObservationId` da sequência canônica. A seleção é explícita:

```python
grouping = group_by_physical_observation(
    observations,
    selected_region_runs=[regional_manifest],
    acquisition_timestamps=timestamps,
)
```

Cada `SpatialObservation` identifica seu `region_artifact_id`, sua observação física e sua região. Claims e features são lidas dos produtores independentes na acumulação; não pertencem à associação espacial. Duas associações para a mesma tripla regional são ambíguas e recusadas.

`PhysicalObservationGrouping` preserva `selected_region_artifact_ids`, um grupo por observação, timestamps e variantes de discovery/refinement. `physical_observation_count` conta frames distintos. Nesta etapa, `inference_result_count` conta envelopes regionais distintos por frame e `perception_run_count` conta produtores regionais que contribuíram. Repetições do mesmo backend/configuração são execuções distintas, mas não variantes distintas.

Na acumulação, os grupos de `FusedEvidence` recebem o ledger de `semantic_interpretation_artifact_ids`. A contagem semântica distingue produtor e observação, mesmo quando uma execução não propõe labels. Três produtores semânticos sobre o mesmo frame continuam correspondendo a uma observação física. Uma execução sobre três frames contribui com três envelopes de inferência.

As vistas derivadas são selecionadas pela chave `(region_artifact_id, source_observation_id)`. A acumulação combina claims semânticas com o suporte regional e deduplica features pela tripla `(visual_artifact_id, source_observation_id, feature_id)`. Repetir uma vista ou compartilhar features entre produtores semânticos não multiplica observações físicas nem claims.

O agrupamento recusa seleção vazia, duplicada ou de sequências diferentes, observações de produtores não selecionados, associações duplicadas/ambíguas e timestamp de aquisição ausente. Sem observações, retorna grupos e contagens vazios. A ordem das entradas não altera o resultado.
