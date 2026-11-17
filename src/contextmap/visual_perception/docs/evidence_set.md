# Seleção explícita de evidência

`PerceptionEvidenceSelection` declara os readers regionais e, opcionalmente, visuais,
semânticos e de scoring que uma view pode combinar. `open(regions=..., visual=...,
semantic=..., scoring=...)` abre somente os diretórios fornecidos pelo chamador.
Não há descoberta automática de runs nem seleção pelo diretório mais recente.

`PerceptionEvidenceSet(selections)` valida sequência comum, seleções únicas e lineage.
Cada produtor visual ou semântico deve nomear o artifact regional selecionado, a mesma
sequência e seleção. Scoring exige os produtores visual e semântico e referencia exatamente
essa combinação. A cobertura dos registros precisa coincidir com os manifests.

`evidence_for(source_observation_id)` retorna `ObservationEvidence.results_by_selection`.
Cada chave nomeia `(region_artifact_id, visual_artifact_id opcional, semantic_artifact_id
opcional, scoring_artifact_ids)`; cada valor é um `PerceptionResult` derivado da seleção.
O resultado conserva os IDs dos produtores e as referências de payload originais.

Esta view não escolhe hipóteses vencedoras, combina confidences, associa objetos entre
runs ou cria entidades persistentes. Inferência repetida sobre a mesma aquisição permanece
evidência correlacionada; essa regra é aplicada por Semantic Fusion.

Os payloads continuam nos artifacts de origem. A view mantém um índice de metadados por
observação física; nenhum artifact é modificado ou reescrito.
