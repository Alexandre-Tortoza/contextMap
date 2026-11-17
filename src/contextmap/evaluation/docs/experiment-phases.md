# Fases de experimento e composições selecionadas (#527)

`contextmap.evaluation.experiment_phases` expande as fases nomeadas do experimento de percepção (#521) em **configurações comuns do runtime**, sem importar nem carregar nenhum modelo. Ele se apoia no manifesto de experimento ([`experiments.md`](experiments.md)), nos braços pareados do #545 e na matriz de capacidades do #522 ([`capability-matrix.md`](capability-matrix.md)). O runtime não pode importar `evaluation`; por isso a camada fica aqui e só chama as funções públicas de resolução do runtime.

## Declaração

| Tipo | Conteúdo |
|---|---|
| `PhaseSpec` | experimento, fase (`PhaseKind`), versão, propósito, estágio avaliado, seleção congelada do reference set, configuração base (overrides do perfil canônico), capacidades e fiação de base, fatores, estágios-alvo, artifacts pinados, composições selecionadas, parâmetros casados, métricas e controles fixos |
| `Factor` | nome, `VariationKind`, níveis e nível baseline |
| `FactorLevel` | overrides `components.*` e `pipeline.stages.*` da configuração do runtime, capacidades que nenhum componente seleciona (uma fonte 3D bloqueada) e arestas produtor → consumidor |
| `MatchedParameters` | parâmetros que todo arm executável dá ao backend selecionado de um componente (política de prompt e de views de Qwen × Eagle 2.5, por exemplo) |

Os nomes dos campos da configuração são os do runtime: `components.semantic_interpretation.interpreter.<backend>.prompt_policy` (com `region_scene_context` para Qwen e Gemini, #529), `...<backend>.view_policy` (`region_views` e os parâmetros de cada view, #524), `...qwen.min_pixels`/`max_pixels` e `components.region_perception.grounding.locateanything.query_set`. A camada não conhece nenhum campo específico: uma política nova entra como override sem mudar o código.

## As onze fases

| `PhaseKind` | Fatores permitidos | Células |
|---|---|---|
| `backend_grid` | `backend` (vários) | um nível por vez |
| `native_task` | `configuration` (um) | um nível por vez |
| `grounding_refinement` | `topology` (um) | um nível por vez |
| `feature_backend` | `backend` ou `evidence_channels` (um) | um nível por vez |
| `semantic_interpreter` | `backend` (um) | um nível por vez |
| `semantic_prompt` | `policy` (um) | um nível por vez |
| `visual_view`, `scene_context` | `evidence_channels` (um) | um nível por vez |
| `visual_budget` | `configuration` (um) | um nível por vez |
| `pipeline_composition`, `downstream_confirmation` | qualquer tipo | só o baseline e as composições **selecionadas**, nunca o produto cartesiano |

Mudança de topologia (inserir o refino, ligar o grounding) é sempre um fator `topology`, distinto de uma troca de backend.

## Expansão

`expand_phase(spec)`:

1. resolve cada arm pelo runtime: `resolve_effective_config` (base + overrides dos níveis), `resolve_plan` (problemas estruturais) e `check_selection` nos componentes dos estágios do arm. Um arm que não resolve **falha a fase**. Os parâmetros de cada backend só são validados na composição, que carrega o adapter; por isso não entram aqui;
2. quando o arm interpreta semântica, resolve sua política de requisição com `resolve_semantic_request_policy` (#544), sem compor backend, e registra `request_policy_fingerprint`; uma política inválida (template de região que não renderiza contexto de cena, `view_policy` ausente) **falha a fase**;
3. monta a topologia do arm: os alvos que ele produz e todos os estágios a montante. Um alvo pode ser um estágio opcional que a configuração do próprio arm desliga (o `semantic_scoring` de um braço "scoring off", #652): esse arm não o produz, e o fator que o liga em outro arm declara a diferença; todo arm produz ao menos um alvo, todo alvo é produzido por ao menos um arm, e um alvo que o arm nem liga nem desliga **falha a fase**. A topologia tem **um nó por ponto de variação selecionado** (`region_perception.refinement`, ...), cuja configuração é o bloco resolvido do componente; um nó depende dos nós dos estágios de entrada, nomeia como `execution` o estágio do runtime que o computa junto com os irmãos, e os estágios pinados levam o artifact imutável;
4. mapeia cada seleção inventariada pela matriz para suas capacidades (o `task` do Florence-2, a `strategy` do SAM3, o `policy_id` de cada query do LocateAnything), mais as operações dos estágios do arm e as capacidades que os níveis declaram. Uma seleção que a matriz não reconhece (SAM3 `automatic`, por exemplo) **falha a fase**, assim como uma capacidade `out_of_scope`, com o motivo da matriz: é o caso das técnicas opcionais que uma decisão deixou fora da campanha (ver [Técnicas opcionais fora da campanha](#técnicas-opcionais-fora-da-campanha-656-657));
5. confere a fiação: uma aresta cujo extremo não está no arm, não declarada, `incompatible` ou `not_scientifically_comparable` **falha a fase**; uma capacidade ou aresta `planned`/`blocked` mantém o arm, `blocked`, com o motivo exato da matriz e a issue (prompt visual do LocateAnything #574, LocateAnything3D #575);
6. confere a aresta **implícita** grounding → associação: um arm com grounding, sem refino e com a associação 2D→3D entre os seus estágios manda as regiões do grounding para a associação mesmo sem declarar a aresta, e ela é verificada na matriz como uma declarada. Hoje ela é `incompatible` (#649) e **falha a fase**, com a nota da matriz (ver [Grounding sem refino não chega à associação](#grounding-sem-refino-não-chega-à-associação-649));
7. deduplica de forma determinística: arms com o mesmo digest de configuração efetiva, a mesma topologia e a mesma fiação ficam `skipped`, com `duplicate_of` apontando o primeiro na ordem das células;
8. confere os parâmetros casados e que uma substituição **só de backend** mantém a tarefa: cada capacidade trocada precisa de uma contraparte do mesmo grupo de comparação (Qwen × Florence-2 `<REGION_TO_CATEGORY>` é recusado);
9. com dois ou mais arms executáveis, monta o `ExperimentManifest` em modo `selected`, cuja construção aplica o #545: qualquer diferença não declarada entre um arm e o baseline falha a fase, assim como uma dependência não pinada de um nó tocado (ver [Pinar pontos de variação, não o estágio](#pinar-pontos-de-variação-não-o-estágio-660)). Os campos de configuração declarados vêm dos overrides de cada fator.

Um baseline bloqueado falha a fase: sem ele nada se compara.

### Braços de scorer semântico (#652)

Os scorers CLIP e AlphaCLIP rodam no estágio opcional `semantic_scoring`, que recebe três artifacts independentes: regional, visual e semântico. Ligar o scorer insere um estágio e é um fator `topology` de composição. A fase declara os alvos visual, semântico e scoring; no braço desligado, os dois primeiros continuam presentes. Todos os produtores de evidência a montante do scorer são pinados, incluindo a região final, para preservar as mesmas claims/features sem repetir a inferência. As arestas de compatibilidade de intérprete e embedding continuam verificadas na matriz. A avaliação distingue scorer e espaço de embedding; sem annotations semânticas, métricas dependentes de referência permanecem N/A ([`semantic-scoring.md`](semantic-scoring.md)).

Uma fase que varia grounding/refinement não pode pinar o artifact regional inteiro: discovery, grounding e refinement são computados dentro desse produtor. Discovery permanece fixado pela identidade da implementação e configuração; cada braço publica suas próprias regiões finais, e os produtores visual/semântico a jusante recebem essas regiões. A topologia explicita a propagação dessa dependência, sem reutilizar um artifact calculado sobre regiões de outro braço.

## Grounding sem refino não chega à associação (#649)

A composição "grounding → associação 2D→3D direta" — P1 da #577, composição B da #528, família B da #521 — **não tem suporte**. A associação define pertencimento pela máscara (`mask-membership-v1`); uma região do grounding só tem caixa (ou é um ponto, que nem vira `Region2D`), e a #649 decidiu não criar uma regra de pertencimento por caixa: a caixa cobre também o fundo e os oclusores, e aceitar os pontos dela esconderia essa contaminação atrás de uma heurística (AGENTS §7). O caminho com suporte é P2: grounding → refino SAM2 em máscara → associação. A justificativa e as arestas estão na [matriz](capability-matrix.md#achados-que-as-fases-seguintes-precisam-saber).

A expansão garante que a campanha não roda P1 em silêncio:

- um arm que **declara** `locateanything.* → contextmap2.mask_membership_association` falha a fase, porque a aresta é `incompatible` e a nota cita a #649;
- um arm que **não declara** a aresta, mas tem grounding, não tem refino e inclui a associação (como alvo ou a montante de um alvo), também falha a fase. Sem a regra, ele resolveria e executaria: a descoberta é obrigatória na composição, então a associação dele teria só as regiões da descoberta, o mesmo conteúdo do P0, sob outro digest de configuração que a deduplicação não reconhece. As caixas apareceriam apenas em `skipped_regions` (`no_inline_mask`) de `outputs/visibility-records.jsonl`, e nenhum relatório diria que o arm não mede o que o nome promete.

A recusa é da fase inteira, como a de qualquer aresta `incompatible`, e não um arm `blocked`: `blocked` significa que uma issue ou um asset vai destravar o arm, e aqui a ausência de suporte é uma decisão de contrato, não uma espera. Por isso P1 nunca entra num `phase.json`, e o relatório da campanha (#528/#577, que ainda não tem executor) precisa listá-lo como não suportado por contrato, com este motivo. Um arm com grounding e refino (P2) segue normalmente, e as caixas originais continuam explícitas em `skipped_regions` com `no_inline_mask`. Um arm com grounding cujos alvos não incluem a associação (a avaliação 2D da caixa, fase 3 da #528) não é afetado.

## Técnicas opcionais fora da campanha (#656, #657)

Três comparações pedidas pela #197 e pela #180 **não entram** na campanha do milestone #22, por decisão registrada antes do freeze dos manifests. A justificativa completa está na [matriz](capability-matrix.md#achados-que-as-fases-seguintes-precisam-saber).

| Comparação | Decisão | Motivo |
|---|---|---|
| features densas nativas × o mesmo artifact DINO → `FeatureResolutionEnhancement` (braço A2 da #197; #180) | #656 | não existe backend concreto do port; escolher e validar um é pesquisa (AGENTS §22), não uma lacuna a fechar |
| Point Representation desligado × determinístico × PTv3, até o mapa final (#180) | #657 | o runtime roda o estágio como saída derivada (#663), mas a fusão e a Entity Resolution não têm entrada `representation`; chegar ao mapa final exige um contrato novo de consumo (pesquisa), e no piloto de ablação o estágio é um controle fixo em todo braço, não uma variável |
| Entity Resolution só com geometria × com evidência 3D de representação (#180) | #657 | o canal `entity_resolution.representation` compara vetores de um run de Point Representation, que o runtime produz (#663) mas o estágio não recebe como entrada; só um provider de vetores fornecido à mão, fora do DAG, o alimentaria |

As três capacidades são `out_of_scope` na matriz (`1.6.0`), com a issue da decisão, e a expansão as recusa assim:

- `contextmap2.feature_resolution_enhancement` não tem identidade no runtime. Um arm só a seleciona declarando-a num nível (ou na base), e a fase falha com o motivo da #656. Um override que tentasse ligar a técnica pela configuração (`pipeline.stages.feature_resolution_enhancement`, `components.visual_perception.feature_resolution_enhancement`) nem resolve: o runtime não conhece o estágio nem o slot;
- `contextmap2.point_representation` se liga ao estágio `point_representation`. Um arm que liga o estágio e tem o próprio estágio como alvo falha a fase com o motivo da #657, qualquer que seja o encoder. Nenhum outro estágio depende dele (a fusão não tem entrada `representation`, #663), então, com outro alvo, o fator que liga o estágio não toca nenhum estágio da fase e ela também falha (#545): um braço nunca sai `planned` como se a representação chegasse ao mapa;
- `contextmap2.entity_representation_comparison` se liga ao ponto de variação `entity_resolution.representation` (backend `entity-representation-comparison-v1`). Um arm que o seleciona falha a fase com o motivo da #657; antes, os dois braços da comparação saíam `planned`, sem motivo, e o braço com o canal só falharia na composição, sem o provider de vetores. Os demais canais da Entity Resolution (semântico, temporal, aparência) seguem na campanha.

Como no #649, a recusa é da fase inteira, não um arm `blocked`: nada vai destravar esses braços nesta campanha. Eles nunca entram num `phase.json`, e o relatório da campanha precisa listá-los como fora da campanha, com a decisão e o motivo: A2 "sem backend", as comparações de Point Representation e da evidência 3D na Entity Resolution "sem consumidor" (o estágio roda desde a #663, mas nada a jusante lê o run). O cenário canônico continua listando as duas opções como só de ablação ([`end-to-end.md`](end-to-end.md#opcionais-só-por-ablação)); a decisão tira as comparações desta campanha, não muda o cenário congelado. O protocolo do experimento A (`build_feature_resolution_protocol`) continua como harness para uma campanha futura que tenha um backend ([`optional-techniques.md`](optional-techniques.md)).

## Toda fase da percepção pina o run de descoberta

A decomposição do DAG da percepção ([`docs/perception-dag-decomposition.md`](../../../../docs/perception-dag-decomposition.md)) tirou a descoberta de regiões para um estágio executado próprio: `region_perception` roda **só** `region_perception.discovery`, publica um `RegionPerceptionRunArtifact` e fica a montante de `visual_perception`, que o consome como entrada `regions`. O grounding e o refino continuam componentes de `visual_perception`: o prefixo da capacidade diz quem é o dono do conceito, a participação no estágio diz onde roda. A consequência para as fases é direta:

- **toda** fase da percepção pina o run de descoberta. Nenhuma fase o varia (o estágio tem um único ponto de variação, e é a descoberta obrigatória), e todo nó de `visual_perception` depende dele. Sem o pin, cada arm descobriria as suas próprias regiões e a comparação ficaria confundida; o #545 recusa a fase nomeando `region_perception.discovery` como dependência não pinada de um nó tocado. Era justamente para isso que a descoberta virou estágio: pinar a parte que não mudou é condição de validade, não otimização;
- o conjunto de pinados continua **declarado**, não derivado. Cada entrada carrega uma `ArtifactIdentity`: um artifact imutável concreto, escolhido por quem monta a fase (AGENTS §8 — a seleção de artifact a montante é explícita). Nenhuma derivação inventa essa identidade. O que **é** derivado — e já era — é a *exigência*: as dependências da topologia vêm de `PlannedStage.inputs`, que vêm do `StageInput.source` do catálogo, e `_check_pinned_upstream` deduz delas quais estágios precisam de artifact pinado. Quando um estágio novo entra na linhagem, a fase falha nomeando-o, em vez de comparar arms com entradas diferentes em silêncio;
- uma fase que liga o grounding e o refino varia o run de **percepção**, não o de descoberta: ela pina `region_perception` e não pode pinar `visual_perception`.

## Pinar pontos de variação, não o estágio (#660)

Os pontos de variação de um estágio do runtime rodam juntos: o run de percepção computa grounding, refino, features e interpretação de uma vez, sobre o run de descoberta pinado. Uma fase que varia um deles (ligar o grounding e o refino, por exemplo) não pode pinar o estágio inteiro, porque o arm variado precisa de outro run; e o #545 exige que toda dependência de um nó tocado seja compartilhada comprovadamente pelos arms. Por isso a expansão pina **pontos de variação**:

- os nós que a fase não declara como variáveis, dentro de um estágio que ela varia, ficam **pinados pela identidade**: rodam de novo em todo arm, e o #545 exige backend e configuração resolvidos iguais em todos (a configuração de `region_perception.discovery` contém `pass_config` e `normalization_config`, de onde saem `RegionDiscoveryIdentity` e a proveniência do backend). Uma mudança num deles sem fator que a declare continua sendo `undeclared change`;
- um estágio que nenhum fator varia continua exigindo um artifact pinado quando está a montante de um nó tocado;
- pinar o estágio que um fator varia **falha a fase**: um artifact imutável é uma computação, e o arm variado reaproveitaria a saída do baseline (pinar `visual_perception` faria o arm P2 reaproveitar o run de percepção do P0, sem grounding).

Assim, a fase P0 × P2 medida na associação (fator `topology` que liga grounding e refino) expande com `ingestion`, `region_perception`, `state_estimation` e `geometric_mapping` pinados, e com as features e a interpretação pinadas pela identidade dentro do run de percepção variado. P1 continua recusado pela #649.

Na execução, os nós pinados pela identidade são estágios não afetados. O runner os compara entre os arms por `kind` + `digest` do `StageArtifact` que o executor reporta, então o executor (#528) precisa reportar para cada um a **própria** saída (para `visual_features.region`, a identidade das features do run). Se reportar o run de percepção inteiro, que difere entre os arms, o par fica `invalid`.

## Política de acumulação da Semantic Fusion (#655)

A matriz inventaria o ponto de variação `semantic_fusion.accumulation` desde a versão `1.6.0`: `baseline-evidence-accumulation-v1` mapeia para `contextmap2.baseline_evidence_accumulation` e `quality-aware-evidence-accumulation-v1` para `contextmap2.quality_aware_evidence_accumulation`, as duas `supported` e no mesmo grupo de comparação, e a aresta `contextmap2.mask_membership_association` → `contextmap2.quality_aware_evidence_accumulation` é `supported`. Antes, a matriz não inventariava esse ponto de variação e a expansão não o mapeava: um arm que selecionava a política ciente de qualidade saía `planned`, sem motivo, e só falhava no preflight, porque o runtime não tinha executor para ela. Agora o `SemanticFusionExecutor` roda as duas políticas, então `planned` volta a significar executável; e, se uma delas deixar de ser executável, a entrada dela na matriz deixa de ser `supported` e o arm sai `blocked` com o motivo e a issue, nunca `planned`.

O experimento B da #197 é uma fase `downstream_confirmation`: fator `fusion_policy` de tipo `backend` (o runtime seleciona a política como backend do ponto de variação), o nível ciente de qualidade com as rampas, o fator neutro e a `definitions_version` nos seus overrides e a aresta acima, e `ingestion`, `visual_perception`, `state_estimation`, `geometric_mapping` e `sensor_association` pinados. Os dois arms consomem o mesmo artifact de associação, e as topologias expandidas alimentam diretamente `build_quality_aware_fusion_protocol` ([`optional-techniques.md`](optional-techniques.md#experimento-b--build_quality_aware_fusion_protocol)).

## Manifesto da fase

`write_phase_manifest(root, manifest)` publica `phase.json` (`contextmap.experiment-phase/v1`), imutável e com digest. Ele guarda a declaração inteira (experimento, fase, fatores e níveis, células, composições, restrições: alvos, pinados, parâmetros casados), o digest da configuração base, a identidade do reference set, a versão da matriz, a identidade do `ExperimentManifest` e, por arm: id, atribuição dos fatores, estado e motivo, issues, digest da configuração efetiva, digest da topologia, identidade da política de requisição semântica, capacidades, fiação e `duplicate_of`. Reexpandir a mesma declaração produz o mesmo documento.

| `ArmState` | Quando |
|---|---|
| `planned` | resolvido e executável |
| `blocked` | usa capacidade ou aresta planejada ou bloqueada; motivo e issues registrados |
| `skipped` | duplicata de outro arm, ou backend indisponível na execução |
| `executed` / `failed` | resultado de `record_phase_outcomes(manifest, run)` depois de `run_experiment`; erro e OOM ficam em `failed` com tipo e mensagem |

`record_phase_outcomes` devolve um novo manifesto (outra identidade) e nunca remove um arm; ele recusa um run de outro experimento.

## Limites

- A validação dos parâmetros de cada backend só ocorre na composição; a expansão garante a resolução, a topologia e a seleção.
- Ainda não existe executor real (#528); a topologia por ponto de variação faz o executor reportar um artifact por nó, mesmo quando vários nós saem do mesmo run de percepção.
- A fiação dentro de um estágio (grounding → refino) é registrada e verificada contra a matriz, mas não vira dependência da topologia: os nós de um estágio só se agrupam pelo run que os computa (`execution`), e nenhum deles pode ser pinado à parte.
