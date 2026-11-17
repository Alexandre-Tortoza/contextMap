# Mapa do código visual

| Arquivo ou pacote | Responsabilidade |
| --- | --- |
| `models.py` | `PerceptionResult`, view derivada com IDs dos produtores |
| `image_preparation.py` | preparação e geometria da imagem |
| `evidence_identity.py` | identidade determinística de features |
| `visual_evidence_artifact.py` | manifest, publicação visual e reader lazy |
| `perception_evidence_set.py` | seleção e agrupamento explícitos de artifacts |
| `ports.py` | fronteira da extração de features |
| `execution/` | contratos, resolução e execução de grafos locais |
| `visual_features/` | features, espaços, payload store, pooling e adapters |
| `feature_diagnostics.py` | métricas contratuais e previews humanos |

Discovery, grounding, refinamento e máscaras residem em `region_perception`.
Requests, prompts, views, parsing e interpreters residem em `semantic_interpretation`.
Scorers e torres de texto residem em `semantic_scoring`. A composição dos adapters e dos
três executores independentes reside em `runtime`.
