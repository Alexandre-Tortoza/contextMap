# Identidade e proveniência

A identidade física vem de `SourceObservationId`, pertencente a Ingestion. Reprocessar
uma observação nunca cria outra aquisição física. Cada artifact produtor preserva essa
identidade e declara `sequence_artifact_id`, `selection_id`, configuração e código.

Regiões pertencem ao run regional. Features pertencem ao run visual; claims e contextos
pertencem ao run semântico. Referências downstream nomeiam esses produtores explicitamente.
`PerceptionResult` reúne a evidência selecionada e não possui identidade própria.

## Features determinísticas

`feature_id_for(run_id, source_observation_id, producer_id, index)` produz a mesma identidade
para as mesmas entradas. Alterar qualquer eixo altera a identidade; produtores de features
podem usar o mesmo índice sem colisão. `VisualFeature.source_observation_id` é obrigatório.

```python
feature_id_for(
    run_id="visual-a",
    source_observation_id=SourceObservationId("frame-0124"),
    producer_id="dense_feature_extraction",
    index=0,
)
```

O namespace de regiões é responsabilidade de `region_perception.namespace_regions`.
A identidade dos requests e claims semânticos é responsabilidade de
`semantic_interpretation`. Nenhum desses IDs identifica automaticamente uma entidade 3D.
Associação espacial, acumulação correlacionada e resolução de entidades continuam explícitas.

Seleções sobrepostas conservam seus produtores e a mesma observação física. Uma feature
ou claim compartilhada por várias views não vira múltiplas medições independentes.
Veja [seleção de evidência](evidence_set.md) e [contratos globais](../../../../docs/CONTRACTS.md).
