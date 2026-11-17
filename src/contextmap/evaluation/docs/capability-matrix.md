# Matriz de capacidades e compatibilidade (#522)

`contextmap.evaluation.capability_matrix` congela, como dados tipados, o que cada família de modelo do experimento do milestone #22 faz nativamente, que papel do ContextMap2 ela ocupa, se já é selecionável pelo runtime e quais composições produtor → consumidor são admissíveis. A versão atual é `1.7.0` (`CAPABILITY_MATRIX_VERSION`); repositórios e model cards upstream foram consultados em 2026-09-25.

A matriz descreve composições experimentais admissíveis; ela **não** escolhe um pipeline canônico. Um modelo só ocupa vários papéis por contratos de evidência explícitos de cada capability: geometria, hipótese semântica e crença persistente nunca se fundem só porque saem da mesma inferência.

## Modelo de dados

| Tipo | Conteúdo |
|---|---|
| `Capability` | uma operação nativa de uma família (ou uma operação do próprio ContextMap2 que a consome): backend, operação nativa, papel e adapter, status, `RuntimeBinding`, evidência de entrada e saída, geometria, controles de prompt/tarefa e de modelo/runtime, campos de proveniência, limitações, papéis compatíveis a montante e a jusante, grupo de comparação, família de métricas, motivo do status e issue |
| `RuntimeBinding` | estágio do preset canônico, componente e backend e os `settings` que selecionam a operação nativa (`task` do Florence-2, `strategy` do SAM3, `query_set.queries.policy_id` do LocateAnything; um caminho pontuado cujas listas se abrem, e a operação é selecionada quando algum valor encontrado é aceito) |
| `Control` | um parâmetro (ou grupo reservado) e seu status; controles ainda não disponíveis citam a issue que os entrega |
| `Composition` | uma aresta produtor → consumidor com status, nota, issue, grupo de comparação e família de métricas |

### Status de uma capability

| Status | Significado |
|---|---|
| `supported` | selecionável hoje pela configuração do runtime: um teste confere que o estágio existe no preset canônico e que o par componente/backend está no catálogo e na composition root |
| `planned` | uma issue aberta do milestone vai torná-la selecionável; se a entrada já nomeia a identidade futura, um teste confere que ela **ainda não** existe (quando existir, a entrada precisa virar `supported`) |
| `blocked` | depende de código ou pesos upstream que não estão publicados |
| `out_of_scope` | fora do experimento: uma operação nativa que ele não usa, ou uma técnica opcional que uma decisão registrada deixou fora da campanha (#656, #657, com a issue da decisão). Não ocupa papel nem grupo de comparação, e nenhuma composição a usa; a expansão de fases recusa o arm que a usa. Quando o runtime ainda seleciona a técnica (o estágio `point_representation`, o canal `entity_resolution.representation`), o `RuntimeBinding` nomeia esse estágio ou ponto de variação, para a expansão reconhecer a seleção e recusar o arm |

### Status de uma composição

| Status | Significado |
|---|---|
| `supported` | o runtime entrega hoje a evidência do produtor ao consumidor |
| `planned` | uma issue do milestone vai entregá-la; até lá é excluída |
| `blocked` | espera assets upstream |
| `incompatible` | o contrato do consumidor não usa a evidência do produtor |
| `not_scientifically_comparable` | executa, mas o resultado não substitui o caminho que parece substituir |

Uma aresta **não declarada** não é admissível. `require_supported_composition(produtor, consumidor)` recusa, antes de carregar qualquer modelo, toda aresta que não seja `supported`, com status, issue e motivo; é o guard que os manifestos do #527 consomem.

## Grupos de comparação

O grupo codifica **semântica da tarefa + condicionamento da entrada + geometria de saída**. `comparable(a, b)` só é verdadeiro no mesmo grupo; uma entrada sem grupo não se compara com nada. Emitir caixa ou máscara nunca basta:

- LocateAnything `category_detection` (caixas condicionadas por categorias da query) e Florence-2 `<OD>` (caixas com rótulo gerado pelo vocabulário de treino) **não** se comparam;
- SAM2 automático (máscaras sem condicionamento) e SAM3 texto (máscaras de todas as instâncias de um conceito) **não** se comparam, nem SAM2 automático com `<REGION_PROPOSAL>` (máscara × caixa);
- SAM3 texto e a composição LocateAnything categoria → refinamento SAM2 compartilham `category_conditioned_instance_masks`: é a comparação de topologia do #521;
- Qwen, Gemini e Eagle 2.5 compartilham os grupos `*_interpretation.instruction_following`, sob a mesma política de prompt, views e schema (#545); as tasks do Florence-2 têm prompt nativo e ficam em grupos `*.task_native`;
- DINOv2, DINOv3 e SigLIP2 denso compartilham `dense_visual_features`; CLIP (crop) e AlphaCLIP (máscara) têm condicionamento diferente e grupos diferentes, assim como os dois scorers, cujas similaridades vivem em espaços distintos.
- as duas políticas de acumulação da Semantic Fusion compartilham `multi_view_evidence_accumulation`: sobre os mesmos suportes elas produzem as mesmas hipóteses, contribuições e incerteza, e a ciente de qualidade só acrescenta o peso por contribuição (experimento B da #197).

Comparar operações de grupos diferentes (por exemplo, caixa × máscara numa métrica de caixa) exige declarar a diferença de tarefa ou geometria como fator do experimento.

## Como o plano do #521 sai da matriz

| Fase do #521 | O que a matriz fornece |
|---|---|
| 1. grade de capability/backend | grupos com dois ou mais membros `supported`: `dense_visual_features`, `scene_interpretation.*`, `region_interpretation.*` |
| 2. tarefa/estratégia nativa | as tasks do Florence-2, a estratégia do SAM3 e as políticas do LocateAnything são capabilities distintas (`RuntimeBinding.settings`); compará-las é ablação de tarefa, não de backend |
| 3. grounding/refinamento | LocateAnything → refinamento SAM2 (`supported`, #568) × SAM3 texto; grounding → associação direta é `incompatible` por decisão (#649): o efeito do refino só se mede em 2D |
| 4. features densas/de região | `dense_visual_features`; amostragem 2D→3D `supported` pelo componente opcional `sensor_association.dense_sampling` (#651), desligado no perfil canônico; features nativas × `FeatureResolutionEnhancement` (#197 A2) fora da campanha por decisão (#656) |
| 5. intérprete semântico | Qwen × Gemini × Eagle 2.5 no mesmo grupo; Florence-2 à parte |
| 6–9. prompt, views, contexto, orçamento | controles `supported`: `prompt_policy`, `view_policy` (#524, em todo intérprete semântico; o Florence-2 só aceita uma view que a região preenche), `prompt_policy.region_scene_context` (#529, Qwen e Gemini; o Eagle 2.5 e o Florence-2 não aceitam contexto de cena) e orçamentos visuais (`min_pixels`/`max_pixels` do Qwen, tiles do Eagle 2.5); a política de requisição resolvida (#544) é a união desses controles, com uma identidade (`SemanticRequestPolicy.fingerprint()`, resolvida só da configuração por `resolve_semantic_request_policy`) |
| 10–11. composições e confirmação downstream | as arestas `supported` até Semantic Fusion, inclusive a acumulação ciente de qualidade (#655); Point Representation desligado × determinístico × PTv3 e Entity Resolution só com geometria × com evidência 3D de representação (#180) fora da campanha por decisão (#657) |

## Achados que as fases seguintes precisam saber

- **Grounding → associação 2D→3D direta não tem suporte (#649).** O pertencimento é definido pela máscara (`mask-membership-v1`): uma região só com caixa (LocateAnything, `<OD>`, `<REGION_PROPOSAL>`, ...) vira `NO_INLINE_MASK`, fica listada com esse motivo em `skipped_regions` do registro de visibilidade do frame e não sustenta nenhum ponto; um ponto do grounding nem vira `Region2D`. A #649 decidiu, antes do freeze dos manifests (#527), **não** criar uma política de pertencimento por caixa: uma caixa não é uma máscara de suporte, porque cobre também o fundo e os oclusores, e aceitar os pontos dela esconderia essa contaminação atrás de uma heurística (AGENTS §7). Por isso a composição "grounding → associação direta" — P1 da #577, composição B da #528, família B da #521 — é declarada sem suporte: as arestas `locateanything.{category_detection,phrase_grounding,pointing}` → `contextmap2.mask_membership_association` são `incompatible` e citam a decisão. O único caminho do grounding até a associação é o refino em máscara (P2 da #577: grounding → SAM2 → associação). A caixa continua avaliável em 2D (fase 3 da #528: grounding → evidência de caixa), e o ganho do refino se mede ali, não como contaminação ou cobertura 3D. A expansão de fases recusa também o braço que, sem declarar a aresta, ligaria grounding sem refino a uma associação (ver [`experiment-phases.md`](experiment-phases.md)).
- **Resolução de features fora da campanha (#656).** O braço A2 da #197 (o mesmo artifact DINO → `FeatureResolutionEnhancement` → associação) e a comparação "features densas nativas × estágio `FeatureResolutionEnhancement`" da #180 **não** entram na campanha do milestone #22. O port, a fronteira de lineage e o protocolo do experimento A existem, mas nenhum backend implementa `enhance()` e o runtime não tem estágio nem slot que o rode. Implementar um backend (LoftUp, FeatUp, JAFAR ou outro) não é fechar uma lacuna: é um projeto de pesquisa, com problema, métrica e avaliação próprios, antes da escolha do modelo (AGENTS §22). A capability `contextmap2.feature_resolution_enhancement` é `out_of_scope` e cita a decisão; um arm que a declara falha a fase. Sem estágio no runtime, uma seleção pela configuração nem resolve.
- **Point Representation fora da campanha (#657).** A comparação "Point Representation desligado × determinístico × PTv3" da #180 **não** entra na campanha. O runtime roda o estágio como saída derivada (#663), mas a fusão e a Entity Resolution não têm entrada `representation`. Levar a evidência até o mapa final exige um contrato novo de consumo na fusão e na Entity Resolution (pesquisa, AGENTS §22), e no piloto de ablação o estágio é um controle fixo em todo braço, não uma variável. A capability `contextmap2.point_representation` é `out_of_scope`, cita a decisão e se liga ao estágio `point_representation`: todo arm que liga o estágio e o tem como alvo falha a fase; com outro alvo, nenhum estágio depende dele e o fator não toca a fase (#545). A mesma decisão cobre a comparação "Entity Resolution só com geometria × com evidência 3D" da #180: o canal `entity_resolution.representation` compara vetores de um run de Point Representation, que o runtime produz (#663) mas o estágio `entity_resolution` não recebe como entrada, então só um provider de vetores fornecido à mão, fora do DAG, o alimentaria. A capability `contextmap2.entity_representation_comparison` é `out_of_scope`, cita a #657 e se liga ao ponto de variação `entity_resolution.representation`: o arm que o seleciona falha a fase. Os demais canais da Entity Resolution (semântico, temporal, aparência) continuam na campanha.
- **Regiões de grounding e refinadas passam pelos canais de região (#650).** Com o grounding composto, o executor roda o preset `grounded/1` (`grounded-refined/1` com refinamento): grounding e refinamento são estágios do grafo de percepção e rodam antes das features de região e da interpretação de região, que consomem o conjunto de regiões do frame (descobertas, caixas de grounding e máscaras refinadas, `region-set-assembly/1`). As arestas LocateAnything → CLIP e → views e refinamento SAM2 → AlphaCLIP, CLIP e views são `supported`. Uma caixa de grounding não tem máscara: o AlphaCLIP e a view `masked_subject` a registram em `outputs/ineligible-regions.jsonl`, sem request nem feature, e as demais regiões do frame seguem.
- **Grounding nunca substitui a descoberta.** `region_discovery` continua obrigatória, então todo braço com grounding mede "descoberta ∪ grounding"; as métricas por fonte filtram por `BackendProvenance.capability` (`region_discovery`, `region_grounding`, `region_refinement`) ou pelo namespace do `region_id`, e as métricas a jusante (fusão, entidades) misturam as fontes. Um braço só com grounding está fora do escopo da #650.
- **As duas acumulações da Semantic Fusion rodam (#655).** O ponto de variação `semantic_fusion.accumulation` é inventariado desde a `1.6.0`: `contextmap2.baseline_evidence_accumulation` e `contextmap2.quality_aware_evidence_accumulation` têm `RuntimeBinding` e são `supported`, porque o `SemanticFusionExecutor` roda as duas; a aresta `contextmap2.mask_membership_association` → `contextmap2.quality_aware_evidence_accumulation` é `supported`, já que o executor lê a `ObservationQuality` que o run de associação persiste. Antes, a política ciente de qualidade era selecionável mas não tinha executor, e um braço que a selecionava expandia como `planned` e só falhava no preflight. `contextmap2.semantic_fusion` continua como a operação do próprio estágio: a construção dos suportes e o agrupamento por observação física.
- **Amostragem densa é opcional e não tem consumidor a jusante (#651).** O run de percepção do runtime grava a geometria de cada mapa denso em `metrics/feature-extraction.jsonl` e, com o componente opcional `sensor_association.dense_sampling` selecionado (`interpolation` = `nearest` ou `bilinear`), o `SensorAssociationExecutor` amostra um canal declarado por feature densa, com o mesmo caminho para DINOv2, DINOv3 e SigLIP2. O perfil canônico continua geometry-only. Semantic Fusion, Semantic Mapping e Entity Resolution não leem as amostras densas: ligar o canal torna mensuráveis a ancoragem e a amostragem na associação, mas não muda a fusão nem o mapa. Os relatórios da #197, da #528 e da #577 precisam dizer isso.
- **Scorers num estágio próprio (#652, desde a `1.6.0`).** `clip.semantic_scoring` e `alphaclip.semantic_scoring` são `supported` no estágio opcional `semantic_scoring`, que lê um `PerceptionRunArtifact` pinado e pontua cada hipótese de região (primária ou alternativa) contra a feature de região da própria região, já extraída naquele run: os braços de scorer compartilham as mesmas claims e o VLM não roda de novo (#580). O CLIP pontua a feature do crop (`ClipRegionSemanticScorer`); a pontuação de cena contra uma feature global continua N/A, porque nenhum slot do runtime produz a feature CLIP global (`qwen.scene_interpretation → clip.semantic_scoring` é `incompatible`). Os espaços CLIP e AlphaCLIP nunca se misturam (as arestas cruzadas são `incompatible`). Como a feature de região é do estágio `visual_perception`, comparar CLIP × AlphaCLIP sobre as **mesmas** claims exige as duas features no mesmo run de percepção, o que um único slot `region_features` não permite; cada braço "off × scorer" usa o seu run pinado. Nenhuma política de fusão consome `SemanticScore`, então o efeito no mapa final segue N/A (#180).
- **SAM3 roda um conjunto de conceitos por run (#662, desde a `1.7.0`).** O controle de prompt de `sam3.text_concept_segmentation` é `prompts`, a lista ordenada de frases que um único run de percepção consulta, cada uma sobre a mesma codificação da imagem de cada passe. Um braço SAM3 produz, assim, um mapa com todos os conceitos do conjunto, comparável no nível do mapa com o SAM2 automático, em vez de um mapa de uma classe só ou de um run por frase acumulado como inferência repetida da mesma imagem. A frase é hipótese de consulta e fica na proveniência de cada proposta; não vira rótulo nem claim.
- **Ponto não vira caixa.** `locateanything.pointing` fica no stream de grounding; não há contrato canônico de ponto.
- **Bloqueados:** LocateAnything com prompt visual (#574; os pesos publicados não suportam) e LocateAnything3D (#575; o repositório público só tem o título do README e não há pesos da NVIDIA publicados).
- **Florence-2 `<OPEN_VOCABULARY_DETECTION>`** devolve caixas ou polígonos conforme a resposta, então sua associação é `not_scientifically_comparable`.

## Fontes consultadas (2026-09-25)

- SAM2: README de `facebookresearch/sam2` (gerador automático de máscaras, predictor de imagem por pontos/caixas, predictor de vídeo).
- SAM3: README de `facebookresearch/sam3` (segmentação por conceito com frase curta ou exemplares, detector e tracker de vídeo, checkpoint com acesso controlado).
- Florence-2: `processing_florence2.py` de `microsoft/Florence-2-large` (lista de tasks e tipo de pós-processamento de cada uma).
- LocateAnything: `NVlabs/Eagle/Embodied/README.md` e o model card `nvidia/LocateAnything-3B` (detecção, grounding de frase, pontos, texto/GUI/layout, modos fast/slow/hybrid, prompt visual sem suporte nos pesos publicados).
- Eagle 2.5: `NVlabs/Eagle/Eagle2_5/README.md` (SigLIP2-So400m-Patch16-512 + Qwen2.5-7B, imagens em alta resolução, até 512 frames de vídeo).
- SigLIP2: documentação do adapter (#572), variantes FixRes e NaFlex.
- LocateAnything3D: `NVlabs/LocateAnything3D` (README só com o título) e busca de modelos no Hugging Face.

## Tabelas

As tabelas abaixo são geradas de `CAPABILITY_MATRIX`; `tests/evaluation/test_capability_matrix.py` falha, mostrando o texto esperado, quando elas divergem dos dados. Controles, evidência, proveniência e limitações de cada entrada estão no módulo.

### Capabilities

<!-- capabilities:start (gerado de CAPABILITY_MATRIX; não editar à mão) -->
| id | backend | operação nativa | papel | status | issue | grupo de comparação |
|---|---|---|---|---|---|---|
| `sam2.automatic_mask_generation` | SAM2 | automatic mask generation (point-grid prompts, IoU/stability filter) | region_discovery | supported | — | automatic_region_proposal.mask |
| `sam2.box_prompt_refinement` | SAM2 | promptable image segmentation from a box or points (image predictor) | region_refinement | supported | — | prompted_mask_refinement |
| `sam2.video_tracking` | SAM2 | video object segmentation with memory (masklets across frames) | — | out_of_scope | — | — |
| `sam3.text_concept_segmentation` | SAM3 | promptable concept segmentation of every instance of a noun phrase | region_discovery | supported | — | category_conditioned_instance_masks |
| `sam3.exemplar_prompts` | SAM3 | concept segmentation from visual exemplars (positive/negative boxes) | — | out_of_scope | — | — |
| `sam3.video_tracking` | SAM3 | concept detection and tracking in video (video predictor) | — | out_of_scope | — | — |
| `florence2.region_proposal` | Florence-2 | <REGION_PROPOSAL> class-agnostic boxes | region_discovery | supported | — | automatic_region_proposal.box |
| `florence2.object_detection` | Florence-2 | <OD> boxes with a generated category name | region_discovery | supported | — | generated_label_detection.box |
| `florence2.dense_region_caption` | Florence-2 | <DENSE_REGION_CAPTION> boxes with a generated description | region_discovery | supported | — | dense_region_caption.box |
| `florence2.open_vocabulary_detection` | Florence-2 | <OPEN_VOCABULARY_DETECTION> boxes or polygons for a text input | region_discovery | supported | — | open_vocabulary_detection.box_or_mask |
| `florence2.caption_phrase_grounding` | Florence-2 | <CAPTION_TO_PHRASE_GROUNDING> boxes for the phrases of a caption | region_discovery | supported | — | caption_phrase_grounding.box |
| `florence2.referring_expression_segmentation` | Florence-2 | <REFERRING_EXPRESSION_SEGMENTATION> polygon mask of a text referent | region_discovery | supported | — | referring_expression_segmentation.mask |
| `florence2.region_to_segmentation` | Florence-2 | <REGION_TO_SEGMENTATION> polygon mask of an input box | region_discovery | supported | — | — |
| `florence2.scene_caption` | Florence-2 | <CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION> scene text | semantic_interpretation | supported | — | scene_caption.task_native |
| `florence2.region_category` | Florence-2 | <REGION_TO_CATEGORY> category of a region | semantic_interpretation | supported | — | region_category.task_native |
| `florence2.region_description` | Florence-2 | <REGION_TO_DESCRIPTION> description of a region | semantic_interpretation | supported | — | region_description.task_native |
| `florence2.ocr` | Florence-2 | <OCR>, <OCR_WITH_REGION>, <REGION_TO_OCR> text reading | — | out_of_scope | — | — |
| `dinov2.dense_patch_features` | DINOv2 | self-supervised patch tokens (CLS and registers removed) | dense_features | supported | — | dense_visual_features |
| `dinov3.dense_patch_features` | DINOv3 | self-supervised patch tokens (CLS and 4 registers removed) | dense_features | supported | — | dense_visual_features |
| `siglip2.dense_patch_features` | SigLIP2 | vision-encoder patch tokens (last_hidden_state, FixRes checkpoints) | dense_features | supported | — | dense_visual_features |
| `siglip2.global_embedding` | SigLIP2 | attention-pooled image embedding (pooler_output) | — | out_of_scope | — | — |
| `contextmap2.feature_resolution_enhancement` | ContextMap2 | raise the grid resolution of one DenseFeatureMap (FeatureResolutionEnhancement) | — | out_of_scope | #656 | — |
| `clip.region_embedding` | CLIP | image projection of a region crop | region_features | supported | — | region_crop_embedding |
| `alphaclip.region_embedding` | AlphaCLIP | image projection conditioned on an alpha (mask) channel | region_features | supported | — | mask_conditioned_region_embedding |
| `clip.semantic_scoring` | CLIP | text-image cosine similarity of region claims and the CLIP crop feature of the same region | semantic_scoring | supported | — | claim_support.clip_region_crop |
| `alphaclip.semantic_scoring` | AlphaCLIP | text-image cosine similarity of region claims and the same region | semantic_scoring | supported | — | claim_support.alphaclip_region |
| `contextmap2.semantic_view_assembly` | ContextMap2 | content-addressed semantic views of the frame or of one region | semantic_view_assembly | supported | — | — |
| `qwen.scene_interpretation` | Qwen | scene interpretation from an instruction and ordered images | semantic_interpretation | supported | — | scene_interpretation.instruction_following |
| `qwen.region_interpretation` | Qwen | region interpretation from an instruction and ordered images | semantic_interpretation | supported | — | region_interpretation.instruction_following |
| `gemini.scene_interpretation` | Gemini | scene interpretation from an instruction and ordered images | semantic_interpretation | supported | — | scene_interpretation.instruction_following |
| `gemini.region_interpretation` | Gemini | region interpretation from an instruction and ordered images | semantic_interpretation | supported | — | region_interpretation.instruction_following |
| `eagle2_5.scene_interpretation` | Eagle 2.5 | scene interpretation from an instruction and ordered images | semantic_interpretation | supported | — | scene_interpretation.instruction_following |
| `eagle2_5.region_interpretation` | Eagle 2.5 | region interpretation from an instruction and ordered images | semantic_interpretation | supported | — | region_interpretation.instruction_following |
| `eagle2_5.long_context_video` | Eagle 2.5 | long-context video and multi-image reasoning (up to 512 frames) | — | out_of_scope | — | — |
| `locateanything.category_detection` | LocateAnything | detect every instance of an ordered category set (PBD boxes) | region_grounding | supported | — | category_conditioned_detection.box |
| `locateanything.phrase_grounding` | LocateAnything | ground every instance matching a free phrase (PBD boxes) | region_grounding | supported | — | phrase_grounding.box |
| `locateanything.pointing` | LocateAnything | point to what a phrase refers to | region_grounding | supported | — | phrase_pointing.point |
| `locateanything.visual_prompt_grounding` | LocateAnything | detection with an image crop as the query (visual prompt) | region_grounding | blocked | #574 | — |
| `locateanything.text_gui_layout` | LocateAnything | scene-text detection, GUI element grounding, layout grounding | — | out_of_scope | — | — |
| `locateanything3d.open_vocabulary_3d_proposal` | LocateAnything3D | open-vocabulary 3D box proposals from images | proposal_3d | blocked | #575 | — |
| `contextmap2.metric_3d_verification` | ContextMap2 | verify a model-predicted 3D proposal against LiDAR/depth geometry | metric_verification | blocked | #575 | — |
| `contextmap2.mask_membership_association` | ContextMap2 | assign visible projected map points to the frozen region masks | sensor_association | supported | — | — |
| `contextmap2.box_component_association` | ContextMap2 | support a box-only region with the 3D component of visible projected map points that its box core sees | sensor_association | supported | — | — |
| `contextmap2.dense_feature_sampling` | ContextMap2 | map each visible point to the cells of a dense feature map | dense_feature_sampling | supported | — | — |
| `contextmap2.point_representation` | ContextMap2 | encode the 3D neighbourhood of selected map points (geometric descriptor or PTv3) | — | out_of_scope | #657 | — |
| `contextmap2.semantic_fusion` | ContextMap2 | build fusion supports over shared geometry, grouped by physical observation | semantic_fusion | supported | — | — |
| `contextmap2.baseline_evidence_accumulation` | ContextMap2 | accumulate every hypothesis of a support, counting distinct physical observations | semantic_fusion | supported | — | multi_view_evidence_accumulation |
| `contextmap2.quality_aware_evidence_accumulation` | ContextMap2 | the baseline accumulation plus a contribution weight derived from the measured observation quality | semantic_fusion | supported | — | multi_view_evidence_accumulation |
| `contextmap2.entity_representation_comparison` | ContextMap2 | compare the Point Representation vectors of two entities | — | out_of_scope | #657 | — |
<!-- capabilities:end -->

### Composições

<!-- compositions:start (gerado de CAPABILITY_MATRIX; não editar à mão) -->
| produtor | consumidor | status | issue | grupo de comparação |
|---|---|---|---|---|
| `locateanything.category_detection` | `sam2.box_prompt_refinement` | supported | — | category_conditioned_instance_masks |
| `locateanything.phrase_grounding` | `sam2.box_prompt_refinement` | supported | — | phrase_conditioned_instance_masks |
| `locateanything.pointing` | `sam2.box_prompt_refinement` | supported | — | — |
| `sam2.box_prompt_refinement` | `contextmap2.mask_membership_association` | supported | — | — |
| `locateanything.category_detection` | `contextmap2.mask_membership_association` | incompatible | — | — |
| `locateanything.phrase_grounding` | `contextmap2.mask_membership_association` | incompatible | — | — |
| `locateanything.pointing` | `contextmap2.mask_membership_association` | incompatible | — | — |
| `florence2.object_detection` | `contextmap2.mask_membership_association` | incompatible | — | — |
| `florence2.object_detection` | `contextmap2.box_component_association` | supported | — | — |
| `locateanything.category_detection` | `contextmap2.box_component_association` | supported | — | — |
| `locateanything.phrase_grounding` | `contextmap2.box_component_association` | supported | — | — |
| `florence2.open_vocabulary_detection` | `contextmap2.mask_membership_association` | not_scientifically_comparable | — | — |
| `sam2.automatic_mask_generation` | `contextmap2.mask_membership_association` | supported | — | — |
| `sam3.text_concept_segmentation` | `contextmap2.mask_membership_association` | supported | — | — |
| `florence2.referring_expression_segmentation` | `contextmap2.mask_membership_association` | supported | — | — |
| `dinov2.dense_patch_features` | `contextmap2.dense_feature_sampling` | supported | — | — |
| `dinov3.dense_patch_features` | `contextmap2.dense_feature_sampling` | supported | — | — |
| `siglip2.dense_patch_features` | `contextmap2.dense_feature_sampling` | supported | — | — |
| `sam2.automatic_mask_generation` | `alphaclip.region_embedding` | supported | — | — |
| `sam3.text_concept_segmentation` | `alphaclip.region_embedding` | supported | — | — |
| `florence2.object_detection` | `alphaclip.region_embedding` | incompatible | — | — |
| `locateanything.category_detection` | `alphaclip.region_embedding` | incompatible | — | — |
| `sam2.box_prompt_refinement` | `alphaclip.region_embedding` | supported | — | — |
| `sam2.automatic_mask_generation` | `clip.region_embedding` | supported | — | — |
| `florence2.object_detection` | `clip.region_embedding` | supported | — | — |
| `locateanything.category_detection` | `clip.region_embedding` | supported | — | — |
| `sam2.box_prompt_refinement` | `clip.region_embedding` | supported | — | — |
| `sam2.automatic_mask_generation` | `contextmap2.semantic_view_assembly` | supported | — | — |
| `florence2.object_detection` | `contextmap2.semantic_view_assembly` | supported | — | — |
| `locateanything.category_detection` | `contextmap2.semantic_view_assembly` | supported | — | — |
| `sam2.box_prompt_refinement` | `contextmap2.semantic_view_assembly` | supported | — | — |
| `contextmap2.semantic_view_assembly` | `qwen.scene_interpretation` | supported | — | — |
| `contextmap2.semantic_view_assembly` | `qwen.region_interpretation` | supported | — | — |
| `contextmap2.semantic_view_assembly` | `gemini.scene_interpretation` | supported | — | — |
| `contextmap2.semantic_view_assembly` | `gemini.region_interpretation` | supported | — | — |
| `contextmap2.semantic_view_assembly` | `eagle2_5.scene_interpretation` | supported | — | — |
| `contextmap2.semantic_view_assembly` | `eagle2_5.region_interpretation` | supported | — | — |
| `contextmap2.semantic_view_assembly` | `florence2.scene_caption` | supported | — | — |
| `contextmap2.semantic_view_assembly` | `florence2.region_category` | supported | — | — |
| `qwen.region_interpretation` | `clip.semantic_scoring` | supported | — | — |
| `qwen.region_interpretation` | `alphaclip.semantic_scoring` | supported | — | — |
| `gemini.region_interpretation` | `clip.semantic_scoring` | supported | — | — |
| `gemini.region_interpretation` | `alphaclip.semantic_scoring` | supported | — | — |
| `eagle2_5.region_interpretation` | `clip.semantic_scoring` | supported | — | — |
| `eagle2_5.region_interpretation` | `alphaclip.semantic_scoring` | supported | — | — |
| `florence2.region_category` | `clip.semantic_scoring` | supported | — | — |
| `florence2.region_category` | `alphaclip.semantic_scoring` | supported | — | — |
| `clip.region_embedding` | `clip.semantic_scoring` | supported | — | — |
| `alphaclip.region_embedding` | `alphaclip.semantic_scoring` | supported | — | — |
| `alphaclip.region_embedding` | `clip.semantic_scoring` | incompatible | — | — |
| `clip.region_embedding` | `alphaclip.semantic_scoring` | incompatible | — | — |
| `qwen.scene_interpretation` | `clip.semantic_scoring` | incompatible | — | — |
| `contextmap2.mask_membership_association` | `contextmap2.semantic_fusion` | supported | — | — |
| `contextmap2.mask_membership_association` | `contextmap2.quality_aware_evidence_accumulation` | supported | — | — |
| `locateanything3d.open_vocabulary_3d_proposal` | `contextmap2.metric_3d_verification` | blocked | #575 | — |
| `contextmap2.metric_3d_verification` | `contextmap2.semantic_fusion` | blocked | #575 | — |
<!-- compositions:end -->
