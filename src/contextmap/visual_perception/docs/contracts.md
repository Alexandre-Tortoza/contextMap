# Contratos visuais

`VisualFeature` registra `feature_id`, `source_observation_id`, scope, embedding space,
shape, dtype, referência de payload e provenance. Features de região referenciam o
`RegionId` final do artifact regional selecionado. Features não contêm labels.

`EmbeddingSpace` define a identidade científica da representação. Dimensão igual não
autoriza comparar espaços distintos. `DenseFeatureMap` declara a transformação entre
imagem e grade, o suporte e o artifact visual produtor; consumers não inferem stride
ou patch size a partir do backend.

`VisualEvidenceResult` agrupa features de uma observação física, inclusive um resultado
vazio. O writer exige que payloads e diagnósticos resolvam exatamente para suas features.
O manifest nomeia o upstream regional e não replica regiões, máscaras nem claims.

`PreparedImage` conserva dimensões, observação, elegibilidade e transformações ordenadas.
Caminhos efêmeros do runtime não são referências duráveis de artifact.

`PerceptionResult` é uma view imutável com `source_observation_id`, `sequence_artifact_id`,
`region_artifact_id`, produtores visual e semântico opcionais e produtores de scoring.
Regiões, features, claims, contexto e scores continuam pertencendo aos respectivos artifacts.
A view valida referências locais e não introduz uma identidade persistida de resultado.

Veja [contratos globais](../../../../docs/CONTRACTS.md),
[artifact visual](visual-evidence-artifact.md) e [identidade](identity.md).
