# Configuração canônica da próxima release (rascunho)

> **Rascunho para a v0.2.0, que não foi lançada.** Este diretório não é o registro de nenhuma release e pode mudar até lá. A próxima release é `minor` porque `[Não lançado]` do [CHANGELOG](../../CHANGELOG.md) já traz transições de schema ([docs/versioning.md](../../docs/versioning.md)).

[`canonical-config.toml`](canonical-config.toml) corrige a configuração canônica congelada da v0.1.0. Aquele arquivo, [`examples/v0.1.0/canonical-config.toml`](../v0.1.0/canonical-config.toml), tem dois defeitos sobrepostos, e continua intacto como registro do que a v0.1.0 entregou:

- **os ids de componente.** Desde a decomposição do DAG de percepção ([docs/perception-dag-decomposition.md](../../docs/perception-dag-decomposition.md)), as quatro chaves da percepção pertencem a outras capabilities (`components.visual_perception.region_discovery` → `components.region_perception.discovery`, e as outras três; a tabela de placeholders abaixo traz os nomes atuais). O arquivo da v0.1.0 nomeia `components.visual_perception`, que o catálogo não conhece mais, então ele é recusado já na resolução — antes que algum backend veja os parâmetros dele. Renomear uma chave ali seria mudar um valor congelado, o que a regra de transição de [`tests/packaging/test_release_examples.py`](../../tests/packaging/test_release_examples.py) não permite;
- **os parâmetros dos backends** (#643). Nenhum backend de percepção aceita os parâmetros dele: SAM2, DINOv2 e CLIP recusam `model` e exigem `checkpoint`, mais `model_version` (SAM2) ou `revision` (DINOv2 e CLIP), e o Qwen exige `precision`, `max_new_tokens` e `temperature`. Desde a ablação de percepção (milestone 22), o Qwen exige também os grupos `prompt_policy` e `view_policy`, sem padrão (#542, #524), que a v0.1.0 não tinha. O defeito está na seção "Erratum" de [docs/releases/v0.1.0.md](../../docs/releases/v0.1.0.md).

Este é um **modelo**, não uma configuração pronta: só estão preenchidos os valores que o repositório registra, e cada valor não registrado é um placeholder que impede a composição.

## O que o teste garante

[`tests/packaging/test_canonical_config_backends.py`](../../tests/packaging/test_canonical_config_backends.py), na instalação base, sem GPU e sem carregar modelo (todo runtime de modelo é um fake):

- com os placeholders preenchidos por valores apenas sintaticamente válidos, os estágios `region_perception` e `visual_perception` são compostos: SAM2, DINOv2, CLIP e Qwen aceitam os parâmetros, cada um pela própria classe de configuração, e a política de request semântico composta é a que reproduz o request da v0.1.0;
- cada placeholder deixado como está, sozinho, impede a composição: o modelo não roda por acidente;
- o arquivo congelado da v0.1.0 é recusado na resolução, pelos ids de componente anteriores à decomposição, e cada bloco dele, transplantado para este modelo sob o id que hoje o possui, é recusado pelo próprio backend (o do Qwen recebe os grupos de política deste modelo, que a v0.1.0 não tinha, para que a recusa seja o defeito do erratum);
- todo placeholder do arquivo está listado abaixo.

O teste cobre só os backends de percepção. Os parâmetros dos demais estágios são decisão do run (abaixo).

## Valores preenchidos e a fonte de cada um

| Parâmetro | Valor | Fonte |
| --- | --- | --- |
| `sam2.checkpoint` | `facebook/sam2.1-hiera-tiny` | tabela "Perfil canônico" de [evaluation/docs/end-to-end.md](../../src/contextmap/evaluation/docs/end-to-end.md) |
| `dinov2.checkpoint` | `facebook/dinov2-base` | idem |
| `clip.checkpoint` | `openai/clip-vit-large-patch14` | idem |
| `qwen.model` | `Qwen/Qwen3-VL-4B-Instruct` | idem |
| `sam2.precision`, `dinov2.precision`, `clip.precision` | `float32` | `"precision": {"sam2_dinov2_clip": "float32"}` em [resource-profile.json](../../experiments/e2e-real-canonical-run-20260923/resource-profile.json) |
| `qwen.quantization` | `4bit` | "Qwen3-VL-4B-Instruct nf4" na seção 3 de [docs/release-v0.1.0.md](../../docs/release-v0.1.0.md); `"4bit"` é o que aplica bitsandbytes NF4 ([Runtime Transformers](../../src/contextmap/visual_perception/docs/semantic-interpretation.md#runtime-transformers-huggingfaceqwenruntime)) |
| `qwen.temperature` | `0.0` | "greedy, `temperature=0.0`" na seção 3 de [docs/release-v0.1.0.md](../../docs/release-v0.1.0.md) |
| `qwen.prompt_policy` | `{ scene = "scene/v1", region = "region/v1" }` | a política canônica, que reproduz byte a byte o prompt anterior à #542 ([Política de prompt semântico](../../src/contextmap/runtime/docs/composition.md#política-de-prompt-semântico-542)) |
| `qwen.view_policy` | `{ region_views = ["tight_crop"] }` | a política equivalente ao comportamento anterior à #524, uma view por região, o recorte justo ([Política de views semânticas](../../src/contextmap/runtime/docs/composition.md#política-de-views-semânticas-524)) |
| `resources.device` | `cuda` | "CUDA (RTX 3060)" em [resource-profile.json](../../experiments/e2e-real-canonical-run-20260923/resource-profile.json); a quantização do Qwen exige CUDA |

Um parâmetro que o arquivo não escreve fica com o default da classe de configuração do backend. Isso não afirma que o run da v0.1.0 usou esse valor.

## Placeholders

Cada um é uma tabela `{ PLACEHOLDER = "<NOME>" }`. Nenhum parâmetro de backend aceita uma tabela, então a composição recusa o backend e nomeia o parâmetro (`BackendConfigurationError`), mesmo onde um texto qualquer passaria: `sam2.model_version` e `qwen.precision` só precisam não ser vazios na validação da configuração. Substitua cada um pelo valor do seu run.

Por que nenhum tem valor registrado: o run real da v0.1.0 gravou a configuração efetiva nos próprios artifacts, e esses artifacts foram podados do workspace local ("Limitações conhecidas" de [docs/releases/v0.1.0.md](../../docs/releases/v0.1.0.md)). Dos parâmetros de Visual Perception, o repositório guardou só o nome dos modelos, a quantização e a temperatura do Qwen, a precisão de SAM2, DINOv2 e CLIP e o device. Inventar um valor gravaria na provenance uma identidade que ninguém verificou.

| Placeholder | Parâmetro | O que o substitui |
| --- | --- | --- |
| `SAM2_MODEL_VERSION` | `components.region_perception.discovery.sam2.model_version` | a versão do checkpoint SAM2 que o seu provider carrega, em texto. Entra na provenance e no digest, e é obrigatória desde a #617: antes, o default `"unknown"` ia para a provenance no lugar da versão real ([region-discovery.md](../../src/contextmap/visual_perception/docs/region-discovery.md#backend-sam2)) |
| `DINOV2_REVISION` | `components.visual_features.dense.dinov2.revision` | o SHA completo, 40 caracteres hexadecimais minúsculos, do commit de `facebook/dinov2-base` no Hugging Face. Uma referência móvel como `main` é recusada, e o loader só lê o cache local por padrão ([dinov2.md](../../src/contextmap/visual_perception/docs/dinov2.md)) |
| `CLIP_REVISION` | `components.visual_features.region.clip.revision` | o SHA completo do commit de `openai/clip-vit-large-patch14`, com as mesmas regras ([clip.md](../../src/contextmap/visual_perception/docs/clip.md)) |
| `QWEN_REVISION` | `components.semantic_interpretation.interpreter.qwen.revision` | o SHA completo do commit de `Qwen/Qwen3-VL-4B-Instruct`. A configuração aceitaria omiti-lo, mas o `HuggingFaceQwenRuntime` recusa `revision=None`, e o cenário canônico só aceita um run que grave a revisão e a quantização ([end-to-end.md](../../src/contextmap/evaluation/docs/end-to-end.md)) |
| `QWEN_PRECISION` | `components.semantic_interpretation.interpreter.qwen.precision` | `float32`, `float16` ou `bfloat16`: o dtype dos pesos não quantizados e o dtype de cômputo da quantização em 4 bits. Os registros dizem "nf4", que é a quantização, não esse dtype. O runtime recusa outro valor só ao carregar o modelo |
| `QWEN_MAX_NEW_TOKENS` | `components.semantic_interpretation.interpreter.qwen.max_new_tokens` | um inteiro positivo, o limite de tokens gerados. Atingi-lo gera um warning, porque o JSON da resposta provavelmente foi truncado ([Runtime Transformers](../../src/contextmap/visual_perception/docs/semantic-interpretation.md#runtime-transformers-huggingfaceqwenruntime)) |

`inputs.sequence` continua um placeholder de texto, como no v0.1.0: é a identidade do seu `SequenceArtifact` já ingerido, não um valor que falte registrar.

## O que este modelo não fixa

**Parâmetros dos demais estágios.** O documento canônico seleciona um backend por ponto de variação e nada mais: "checkpoint, revisão e limiares pertencem a quem os possui e ao run que os grava; o documento não os fixa" ([end-to-end.md](../../src/contextmap/evaluation/docs/end-to-end.md), "O perfil como configuração do runtime"). Um run real precisa acrescentar, no bloco de cada backend, os parâmetros que a composição exige:

| Componente | Parâmetros exigidos |
| --- | --- |
| `state_estimation.estimator` (`external_pose`) | `reference_frame`, `body_frame` |
| `geometric_mapping.pose_lookup`, `sensor_association.pose_policy` (`lookup-policy-v1`) | `mode` |
| `geometric_mapping.motion_correction` | `raw`, `unknown` |
| `sensor_association.occlusion` | `cell_size_px`, `neighborhood_radius_cells`, `depth_margin_m`, `depth_margin_ratio` |
| `sensor_association.tolerances` | `max_pose_time_delta_ns`, `max_map_window_offset_ns`, `max_reprojection_p95_px`, `max_reprojection_invalid_rate` |
| `semantic_fusion.support` | `min_geometry_count`, `min_overlap` |
| `semantic_mapping.geometry_summary`, `spatial_relations.geometry_summary` | `sparse_point_threshold`, `connectivity_radius_m` |
| `entity_resolution.retrieval` | `centroid_radius_m`, `bounds_margin_m` |
| `entity_resolution.resolution` | `use_channels`, `min_supporting_channels` |
| `entity_resolution.geometry_comparison` | `min_shared_support_jaccard`, `min_bounds_iou`, `min_bounds_containment`, `min_conflict_gap_m`, `min_extent_ratio` |
| `spatial_relations.frame_conventions` | `map_frame` |
| `spatial_relations.candidate` | `predicates`, `proximity_radius_m`, `directional_radius_m` |

Frames e modos dependem do dataset; limiares são decisão científica do run e ficam gravados nele. Na v0.1.0, os limiares de semantic mapping, entity resolution e spatial relations foram reaproveitados do fixture sintético de CI, não derivados de dados reais ("Limitações conhecidas" de [docs/releases/v0.1.0.md](../../docs/releases/v0.1.0.md)). Formato, camadas e validação da configuração: [runtime/docs/configuration.md](../../src/contextmap/runtime/docs/configuration.md); o que cada componente constrói: [runtime/docs/composition.md](../../src/contextmap/runtime/docs/composition.md).

**Runtimes de SAM2 e Qwen.** Nenhum dos dois tem loader empacotado na composição: o runtime vem de um `RuntimeProvider`, passado pela API Python (`providers=`) ou declarado em `[resources.providers]` para o `contextmap` instalado ([composition.md](../../src/contextmap/runtime/docs/composition.md#providers-declarados-em-configuração-resourcesproviders)). DINOv2 e CLIP usam o loader Hugging Face empacotado, que exige o extra `vision`.

## Usar

```bash
contextmap inspect config -c examples/v0.2.0/canonical-config.toml --json
contextmap inspect plan   -c examples/v0.2.0/canonical-config.toml
```

Os dois resolvem o modelo como está: para o documento, um placeholder é só um valor. Para um run, copie o arquivo, substitua cada placeholder e `inputs.sequence`, acrescente os parâmetros dos demais estágios e os providers de SAM2 e Qwen. Enquanto restar um placeholder, a composição da percepção falha nomeando o parâmetro.

## Documentos gerados

Dois documentos ao lado do modelo são **gerados, não escritos à mão**, e são os que [`tests/packaging/test_release_examples.py`](../../tests/packaging/test_release_examples.py) compara com o que este código resolve:

| Arquivo | O que é |
| --- | --- |
| [`effective-config.json`](effective-config.json) | a configuração **efetiva** resolvida a partir do modelo, com o digest e as fontes |
| [`resolved-plan.txt`](resolved-plan.txt) | a topologia resolvida, sem executar nada |

O teste só aceita que o código resolva outro digest com a transição registrada em `[Não lançado]` do [CHANGELOG](../../CHANGELOG.md) (`canonical effective configuration <digest congelado> → <digest do código>`), e só por acréscimo: todo valor do documento efetivo e toda linha do plano congelados precisam continuar resolvendo iguais. Regenerar, da raiz do repositório:

```bash
contextmap inspect config -c examples/v0.2.0/canonical-config.toml --json > examples/v0.2.0/effective-config.json
contextmap inspect plan   -c examples/v0.2.0/canonical-config.toml       > examples/v0.2.0/resolved-plan.txt
```

O artifact de demonstração da v0.1.0 ([`examples/v0.1.0/demo`](../v0.1.0/demo)) não muda e continua sendo o que `contextmap validate` e `contextmap inspect artifact` abrem nos testes de release.
