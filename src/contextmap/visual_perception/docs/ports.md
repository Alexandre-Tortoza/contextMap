# Ports visuais

`FeatureExtractor` é a fronteira substituível de extração. Declara provenance e scope;
`extract(image, regions=())` retorna metadados `VisualFeature` sem tensores nativos.
Adapters densos, globais e regionais preservam observação, suporte, embedding space e dtype.

O sink entrega `add_feature_payload`, `add_feature_diagnostic` e `add_feature_preview` ao
writer visual. O payload é persistido imediatamente em staging; sua correspondência com
features e geometria é validada antes da publicação.

Regiões vêm da API pública regional. Resolver máscaras é uma dependência explícita;
caixas não viram máscaras silenciosamente. `release()` libera os recursos do backend
no ciclo de vida coordenado pelo runtime.

Os ports de discovery/grounding/refinamento pertencem a Region Perception; os de
interpretação e scoring pertencem aos respectivos produtores semânticos.
