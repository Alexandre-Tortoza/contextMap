# Campanha de ablação de capacidades no corridor-02 2:30–2:35, um mapa por braço (issue #670)

A campanha compara as capacidades do projeto sobre um mesmo trecho real: cada braço varia um eixo a partir do baseline, e os artifacts de cada braço entram no índice do viewer externo assim que o braço termina. Todos os braços partem da mesma `SpatialFoundation` (sequência, trajetória e mapa geométrico) e da mesma seleção de imagens. O piloto ([#665](../perception-pilot-ablation-20260927/README.md)) roda antes, como gate de ambiente e de providers. Não há reference set (#581): a campanha mede diferenças, custo e repetibilidade, não qualidade contra verdade anotada. O protocolo foi registrado na #670 antes de qualquer corrida e está copiado literalmente no [anexo](#anexo-protocolo-registrado-cópia-literal).

> **Estado: preparado, ainda não executado em dado real.** O harness, a matriz, as camadas de configuração, as verificações e as métricas estão implementados e testados; um dry run sobre a cadeia de CI percorre os 14 braços pela CLI na suíte (`tests/end_to_end/test_perception_ablation_campaign_harness.py`). Este ambiente não tem GPU, checkpoints nem o corridor-02: não há número real a citar até o comando abaixo rodar na máquina com GPU e o resultado ser registrado na seção [Resultado](#resultado).

## Braços

Os braços rodam nesta ordem, a de [`config/campaign.json`](config/campaign.json): o baseline primeiro, os braços que reusam a percepção dele logo depois (são baratos e chegam cedo ao viewer), os de percepção completa, a réplica e, por último, o Eagle 2.5. A rodada é fail-fast, e o Eagle 2.5 é o braço que mais pode falhar na máquina real (8B em `bfloat16`, sem quantização): por último, uma falha dele não custa nenhum outro braço, e a concordância da réplica é gravada assim que a réplica termina.

| Braço (camada em `config/arms/`) | Grupo | Varia | Execução |
|---|---|---|---|
| `sam2-region-v1` | discovery × prompt de região | baseline | percepção completa |
| `fusion-quality-aware` | fusão | `quality-aware-evidence-accumulation-v1` | build extra sobre a branch do baseline |
| `association-dinov3-dense` | canal denso na associação | `dense-feature-sampling-v1`, `bilinear` | associação e jusante sobre a percepção do baseline |
| `scoring-clip` | scoring semântico | estágio `semantic_scoring` com CLIP | scoring sobre a percepção do baseline |
| `sam2-region-abstention-v1` | discovery × prompt de região | `region-abstention/v1` | percepção completa |
| `sam2-region-scene-context-v1` | discovery × prompt de região | `region-scene-context/v1` | percepção completa |
| `sam3-region-v1` | discovery × prompt de região | SAM3 com conjunto de conceitos | percepção completa |
| `sam3-region-abstention-v1` | discovery × prompt de região | SAM3, `region-abstention/v1` | percepção completa |
| `sam3-region-scene-context-v1` | discovery × prompt de região | SAM3, `region-scene-context/v1` | percepção completa |
| `interpreter-florence2-region-category` | intérprete | Florence-2 `<REGION_TO_CATEGORY>` | percepção completa |
| `qwen-view-contextual-crop` | views do Qwen | `contextual_crop` | percepção completa |
| `qwen-view-masked-subject` | views do Qwen | `masked_subject` | percepção completa |
| `sam2-region-v1-replica` | discovery × prompt de região | réplica do baseline, mede o ruído | percepção completa |
| `interpreter-eagle2-5` | intérprete | Eagle 2.5 | percepção completa |

As sete camadas do piloto são as do piloto, byte a byte, e a base e a camada do PTv3 são as dele (um teste confere): o baseline da campanha é o baseline do piloto.

## Como o harness implementa o protocolo

| Protocolo | Implementação |
|---|---|
| Trecho de 2:30 a 2:35 ingerido pela #669 | `--window-file`: o `window.json` que a #669 grava (`contextmap.corridor-window/v1`). O harness usa só o schema, `sequence_artifact_id`, `clock_id`, `start_ns`, `end_ns` e `quality.within_thresholds`; os outros campos, inclusive os que a #669 acrescentou ao contrato (`start_recording_ns`, `canonical_sequence_artifact_id`, ...), são aceitos e gravados como vieram em `protocol.json` |
| Uma imagem a cada 6 frames da câmera | da sequência da fundação, cada 6ª imagem cujo timestamp está em `[start_ns, end_ns)`, a primeira incluída, na ordem do índice canônico (`selection.image_stride` da matriz). Os ids explícitos entram em todo `context run` como `inputs.observation_selection` do tipo `explicit_ids` e ficam em `protocol.json` |
| Uma única `SpatialFoundation` | `--foundation-run`: um run concluído cuja execução nomeia sequência, trajetória e mapa, o mesmo que `contextmap context branch create --from-run` aceita |
| Baseline e braços como dado | [`config/campaign.json`](config/campaign.json): as variáveis, e por braço o papel (`baseline`, `arm`, `replica`), o grupo, o nível de cada variável que ele muda, a camada, a execução e o braço cuja percepção ele reusa; e os parâmetros dos eixos novos com a origem de cada um |
| Configuração base e camadas | [`config/base.template.json`](config/base.template.json) (a do piloto), [`config/point-representation.template.json`](config/point-representation.template.json) por `--all-arms-config` (PTv3 ligado em todo braço, como no piloto) e uma camada por braço em [`config/arms/`](config/arms/) |
| Reuso dos passos do #501 e dos módulos do piloto | `run_campaign.py` põe os diretórios de scripts do #501 e do piloto no caminho de import: o comando medido, `StepFailed`, o registro do passo e do build, a configuração sem caminho da máquina e a regra da identidade do código são os do #501; as verificações das camadas (`check_controlled`, `refuse_placeholders`, `resolve_arm`, `arm_layers`) e as métricas (`perception_metrics`, `replica_agreement`) são as do piloto, que passam a receber os braços, as variáveis e o baseline como parâmetro (o piloto continua com os dele por padrão) |
| Camadas voxel | antes dos braços, a parte geométrica do [`run_all.py`](../geometric-aggregation-raw-vs-voxel-20260926/scripts/run_all.py) da #624 (`--raw-map <mapa da fundação> --cells 0.05 0.10`, sem `--sequence`, `--trajectory` nem `--perception`) num processo filho medido, em `harness/voxel-layers/`; falha como qualquer passo |
| Índice do viewer | assim que um braço termina: `harness/maps/<braço>.json`, publicado uma vez, e `harness/maps.json` regravado, os dois escritos num arquivo temporário do mesmo diretório e renomeados no lugar; um leitor vê o índice anterior ou o novo, inteiro |
| Fail-fast, nunca sobrescrever | o primeiro passo que falha encerra a rodada com `report.json` `failed`; um diretório de rodada existente é recusado |

### Os braços de reuso

Cada braço de reuso roda por um caminho que a runtime suporta sem reinferir nada (o Qwen não é determinístico: uma percepção refeita seria outra evidência):

- **`fusion-quality-aware`** — `contextmap context build` sobre a branch do baseline, revisão 1, com a camada do braço: o padrão do cenário 3 da #501. O build recebe a percepção e a associação da `ContextRun` do baseline e só roda da fusão em diante. O braço não tem branch própria; a entrada dele no índice cita a branch do baseline.
- **`association-dinov3-dense`** — `contextmap run --stage context_map --stage point_representation`, com a sequência, a trajetória e o mapa da fundação e a percepção do baseline selecionados (`--select`) de um catálogo explícito que o harness escreve (`harness/catalogs/<braço>.json`, [`selection.md`](../../src/contextmap/runtime/docs/selection.md#arquivo-de-catálogo)): associação, fusão, mapeamento, resolução, relações, mapa e PTv3 são refeitos; a percepção não. O braço **não tem branch**: `contextmap context run` sempre executa a percepção (o escopo de uma `ContextRun` só fornece a fundação), `--reuse-index` exige um verificador que o binário instalado não tem, e `--select` num `context run` é ignorado em silêncio (ver [Limitações](#limitações-declaradas)). O mapa é materializado por um run comum da runtime, sem `ContextBuild`; o relatório lê dele a identidade, as contagens, a validação `FULL` e a integridade (`materialization`), pelos mesmos leitores públicos.
- **`scoring-clip`** — `contextmap stage semantic_scoring` com a percepção do baseline selecionada de um catálogo explícito. O scoring não produz mapa próprio: nenhuma política de fusão consome `SemanticScore` (#180), então a entrada do braço no índice cita o mapa, a branch e os runs do baseline, mais o run de scoring.

Cada braço de reuso grava em `report.json` (`arms.<braço>.reuse`) a referência exata (estágio, contrato, id, content hash e local) do artifact do baseline e a do artifact que o run consumiu, lida do registro de execução do próprio run (`execution.json` ou `context_build.json`), e a disposição de cada estágio: a percepção é `provided`, nunca executada. Se o artifact consumido não for o do baseline, a rodada para no passo `reuse-check`.

Os comandos de reuso só carregam os providers dos estágios que executam: a CLI compõe todo estágio que a configuração permite, e um provider carrega o modelo ao ser composto. O harness passa `--set 'resources.providers={...}'` com os providers dos componentes dos estágios do run (o do PTv3 no braço denso, nenhum no build e no scoring, cujo CLIP de texto tem loader empacotado). É a regra do build do piloto, generalizada.

### Onde o protocolo não fixa o detalhe

Cada escolha fica registrada em `protocol.json`:

1. **Nome das branches.** `campaign-run-NNNN-<braço>`, pela convenção do #501 e do piloto: uma rodada refeita com `--run-number` seguinte no mesmo workspace não encontra as branches da anterior.
2. **Variáveis.** Os eixos novos são variáveis da matriz, verificadas par a par como no piloto (`contextmap.evaluation.arm_differences`, campo a campo, sobre a configuração resolvida; `resources` fica de fora, é implantação). O scoring é declarado por duas variáveis que o braço atribui juntas, porque muda duas coisas: `semantic_scoring_stage` liga o estágio em `pipeline.stages` (uma configuração) e `semantic_scorer` insere o componente `visual_perception.semantic_scoring` (uma topologia).
3. **Parâmetros registrados e espaço do scorer.** Cada valor da tabela da issue está em `campaign.json` (`parameters`) com o caminho na configuração e a origem; antes de criar qualquer coisa o harness confere que o braço resolve exatamente esse valor, e recusa a rodada se não resolver (a origem registrada seria de um valor que ninguém usou). Do mesmo jeito, o scorer do braço de scoring precisa nomear o backend, o `checkpoint` e a `revision` de `region_features` do baseline, cujas features ele pontua; sem isso a recusa só viria na execução do estágio, depois dos braços anteriores (`protocol.json`, `scorer_space`).
4. **`observation-quality-v3`.** O `s04_fusion.py` do experimento de 2026-09-21, de onde vêm as rampas e o fator neutro, declarou `observation-quality-v1`; a associação atual grava `observation-quality-v3` e a fusão ciente de qualidade recusa outra versão. A campanha usa a v3 (a v3 só redefiniu os componentes por pixel sobre o footprint da região; as rampas de profundidade e de borda seguem as mesmas), e a origem registrada diz isso.
5. **`contextual_crop`.** Só `context_margin_ratio` (0,5) é registrado; `context_boundary` fica ausente, isto é, nenhum contorno é desenhado. O exemplo de `composition.md` de onde vem a margem também desenha um contorno vermelho de 2 px, que a issue não pede.
6. **Florence-2.** A task `<REGION_TO_CATEGORY>` serve só o modo `region` (`supported_modes: ["region"]`) e aceita uma view que a região preenche (`tight_crop`), sem `prompt_policy`. O preset canônico da percepção ainda pede um request de cena por frame: sem prompt de cena composto, o estágio `scene_interpretation` falha em todo frame, gravado como resultado de estágio `failed`, e o braço não tem claim de cena. As regiões são interpretadas normalmente.
7. **Eagle 2.5.** `nvidia/Eagle2.5-8B` com `max_dynamic_tiles` 6, `temperature` 0 e a mesma `prompt_policy` e `view_policy` do baseline. `EagleSemanticConfig` não tem quantização: com os pesos em `bfloat16` o modelo de 8B não cabe na RTX 3060 de 12 GB citada nos custos do piloto; a máquina precisa de mais memória ou de um provider que resolva isso. Por isso ele roda por último.
8. **Seleção de uma janela.** Uma janela de outra sequência, com as imagens noutro clock, sem imagem ou fora dos limiares de lacuna é recusada antes de criar qualquer coisa. `quality.within_thresholds` falso (inclusive quando uma lacuna não pôde ser medida, `null`) só roda com `--accept-window-gaps`, que o protocolo grava.

## Métricas

Todas por braço, lidas pelos leitores públicos dos artifacts; nenhuma é combinada com outra, e nenhuma é afirmação de qualidade.

| Métrica | Onde, em `report.json` | Definição |
|---|---|---|
| status e falhas | `arms.<braço>.status`, `failure` | `complete`, `failed` (o braço em que a rodada parou) ou `not_run`; a falha nomeia braço, passo, mensagem da CLI e dica |
| percepção | `arms.<braço>.perception` | as métricas do piloto (regiões por frame, requests semânticos, falhas de chamada, abstenção por modo, labels); nos braços de reuso, `reused_from` |
| build | `arms.<braço>.build` | o registro do #501 (identidades, contagens, validação `FULL`, integridade, bytes por artifact) dos braços com build |
| materialização sem build | `arms.association-dinov3-dense.materialization` | identidade do mapa, contagens da fusão (como gravadas), entidades, entidades resolvidas e relações por estado, validação `FULL`, integridade e bytes |
| scoring | `arms.scoring-clip.scoring` | o que o run pontuou: resultados, claims de região e de cena, claims pontuadas e não pontuadas, scores, espaço, template e integridade |
| reuso | `arms.<braço>.reuse` | a referência do artifact do baseline e a do consumido, se são o mesmo e a disposição de cada estágio |
| tempo e memória | `arms.<braço>.steps.<passo>`, `voxel_layers.step` | tempo de parede e pico de RSS do processo de cada comando |
| concordância baseline × réplica | `replica_agreement` | a do piloto: `pilot-replica-join/v1`, `min_iou` 0,5, só pares rotulados dos dois lados |
| critério | `criterion` | 14 braços completos, 13 mapas verificados e distintos (o scoring não tem mapa próprio) e os 3 braços de reuso com o reuso comprovado |

## Pré-requisitos na máquina com GPU

**Gates.** O piloto (#665) rodou e passou; a #669 gravou o `window.json` e a fundação sobre a sequência da janela, com a trajetória ground truth pela ponte de pose (`policies.trajectory_mode = "allow_ground_truth"`, já na base) e o mapa geométrico bruto. O harness não cria a fundação; ver "Fundação" no [README do #501](../incremental-partition-invariance-20260927/README.md#pré-requisitos-na-máquina-com-gpu).

**Ambiente.** O do piloto: checkout limpo no commit que será citado, `pip install -e '.[vision]'`, o ambiente do Pointcept para o PTv3 e os providers declarados. Um run real é recusado com status 2, antes de criar qualquer coisa, quando a árvore não está comprovadamente limpa, quando nenhum commit identifica o checkout ou quando `--code-identity` não nomeia exatamente o commit do checkout (`resolve_code_identity`, a regra do #501); o dry run é isento. O Eagle 2.5-8B roda em `bfloat16`, sem quantização: confira a memória da GPU antes da rodada; uma falha dele, último braço, deixa publicados os mapas de todos os outros.

**Configuração.** Copie `config/` para fora do Git e preencha cada placeholder da cópia; o harness recusa a rodada enquanto restar um. Os placeholders da base, das camadas SAM3 e do PTv3 são os do piloto ([tabela](../perception-pilot-ablation-20260927/README.md#pré-requisitos-na-máquina-com-gpu)); os novos:

| Placeholder | Arquivo | O que o substitui |
|---|---|---|
| `REPLACE-WITH-THE-SEQUENCE-ARTIFACT-ID` | base | a identidade do `SequenceArtifact` da janela, o `sequence_artifact_id` do `window.json` |
| `REPLACE_WITH_YOUR_LOADERS_MODULE:load_eagle2_5` | `interpreter-eagle2-5.json` | o provider do Eagle 2.5 (`HuggingFaceEagleRuntime(config=config)`, [`eagle2_5.md`](../../src/contextmap/visual_perception/docs/eagle2_5.md)) |
| `REPLACE-WITH-THE-40-HEX-EAGLE-REVISION` | `interpreter-eagle2-5.json` | o SHA completo do commit de `nvidia/Eagle2.5-8B` (o de `main` em 2026-09 era `61e45235a1e4a35222b86383479b3b004f4809d7`, nunca executado aqui) |
| `REPLACE-WITH-THE-EAGLE-PRECISION`, `REPLACE-WITH-THE-EAGLE-TOKEN-BUDGET` | `interpreter-eagle2-5.json` | o dtype dos pesos e o `max_new_tokens`; o orçamento de tokens deve ser o do Qwen |
| `REPLACE_WITH_YOUR_LOADERS_MODULE:load_florence2_semantic` | `interpreter-florence2-region-category.json` | o provider do Florence-2 semântico (`Florence2SemanticRuntime`) |
| `REPLACE-WITH-THE-40-HEX-FLORENCE2-REVISION`, `REPLACE-WITH-THE-FLORENCE2-PRECISION`, `REPLACE-WITH-THE-FLORENCE2-TOKEN-BUDGET` | `interpreter-florence2-region-category.json` | a revisão de `florence-community/Florence-2-large`, o dtype e o `max_new_tokens` |
| `REPLACE-WITH-THE-40-HEX-CLIP-REVISION` | `scoring-clip.json` | **a mesma** revisão do CLIP de `region_features` da base: o scorer só compara no espaço de embedding das features de região do run de percepção. O harness recusa a rodada, antes de criar qualquer coisa, se o backend, o `checkpoint` ou a `revision` do scorer forem outros que os de `region_features` na configuração resolvida do baseline |

**Custo.** O do piloto por braço de percepção (~37 s/frame na RTX 3060 para a percepção do v0.1.0); com ~20 imagens, 11 braços de percepção completa. Os braços de reuso não rodam modelo de percepção.

## Comando

Da raiz do checkout, depois de copiar e preencher a configuração:

```bash
cp -r experiments/perception-ablation-campaign-20260927/config outputs/perception-ablation-campaign-config
# preencher cada REPLACE... da cópia (tabelas acima e do piloto)

python experiments/perception-ablation-campaign-20260927/scripts/run_campaign.py \
    --workspace outputs/<workspace> \
    --foundation-run outputs/<workspace>/<dataset>/run-NNNN \
    --window-file <window.json da #669> \
    --base-config outputs/perception-ablation-campaign-config/base.template.json \
    --arms-dir outputs/perception-ablation-campaign-config/arms \
    --all-arms-config outputs/perception-ablation-campaign-config/point-representation.template.json
```

Padrões: `--code-identity` é o `HEAD` do checkout, gravado como SHA completo; `--run-number 1` numera a rodada; `--accept-window-gaps` só se a janela estiver fora dos limiares e o mantenedor decidir rodá-la assim. A matriz é sempre a do checkout (`config/campaign.json`), nunca a da cópia: ela não tem valor de máquina.

Por braço, o harness executa comandos da CLI instalada, cada um num processo novo e medido, pelos passos do #501:

```bash
# percepção completa
contextmap context branch create campaign-run-NNNN-<braço> --from-run <fundação> ...
contextmap context run --branch campaign-run-NNNN-<braço> --code-identity <id> --set 'inputs.observation_selection={...}' ...
contextmap context build --branch campaign-run-NNNN-<braço> --revision 1 --code-identity <id> --set 'resources.providers={}' ...
# fusion-quality-aware
contextmap context build --branch campaign-run-NNNN-sam2-region-v1 --revision 1 --code-identity <id> -c ... -c fusion-quality-aware.json --set 'resources.providers={}'
# association-dinov3-dense
contextmap run --stage context_map --stage point_representation --select ingestion=<id> --select state_estimation=<id> --select geometric_mapping=<id> --select visual_perception=<percepção do baseline> --catalog harness/catalogs/association-dinov3-dense.json --code-identity <id> --set 'resources.providers={"point_representation.encoder": ...}'
# scoring-clip
contextmap stage semantic_scoring --select visual_perception=<percepção do baseline> --catalog harness/catalogs/scoring-clip.json --code-identity <id> --set 'resources.providers={}'
```

## O que o comando grava

Em `outputs/perception-ablation-campaign-20260927/run-NNNN/harness/` (fora do Git):

| Arquivo | Conteúdo |
|---|---|
| `protocol.json` | gravado **antes** do primeiro comando: commit e estado da árvore, identidade do código, a matriz (arquivo e SHA-256), a fundação, a janela (arquivo, SHA-256, o registro inteiro e `accept_window_gaps`), a seleção (passo, imagens na janela, ids), as variáveis, os parâmetros com a origem, e por braço o papel, o grupo, as atribuições, a camada, a execução, o reuso, a branch, a configuração efetiva e as diferenças em relação ao baseline; as camadas voxel, a política de join da réplica e a definição dos labels |
| `report.json` | por braço: status, passos, métricas, build ou materialização, scoring e reuso; as camadas voxel; a concordância da réplica; o critério; a falha, quando houver |
| `maps.json` | o índice do viewer: por braço, na ordem em que terminaram, branch (ou `null`), `context_map_id` e os caminhos, relativos ao workspace, do mapa, da percepção, da associação, da fusão, do PTv3 e do scoring; e as camadas voxel por tamanho de célula, relativas a `harness/` |
| `maps/<braço>.json` | a entrada de cada braço no índice, publicada uma vez |
| `voxel-layers/` | a saída do `run_all.py` da #624: o relatório e um `VoxelAggregationArtifact` por tamanho de célula |
| `catalogs/` | o catálogo explícito de cada braço que seleciona artifacts |
| `steps/<braço>/` | o pedido e a resposta de cada comando e, no passo que falhou, o fim do stderr; têm caminhos da máquina |

`protocol.json`, `report.json` e `maps.json` não gravam caminho absoluto da máquina: a mensagem de uma falha e os textos do registro da janela trocam o workspace, a raiz de saída, o checkout e o home por `<workspace>`, `<outputs>`, `<checkout>` e `~`.

## O que volta para o Git

Depois da corrida real, numa árvore limpa, no commit citado:

1. Copiar para este diretório `protocol.json`, `report.json` e `maps.json` (não `steps/`, `voxel-layers/`, `catalogs/` nem os artifacts do workspace).
2. Escrever o `manifest.json` do bundle com o SHA-256 completo de cada arquivo executado (`scripts/`, `config/`) e de cada registro copiado ([`docs/ARTIFACTS.md`](../../docs/ARTIFACTS.md)); a partir daí `tests/evaluation/test_experiment_bundles.py` confere o bundle. As cópias preenchidas da configuração não voltam: a configuração efetiva de cada braço já está em `protocol.json`.
3. Preencher a seção [Resultado](#resultado).

## Resultado

> **Pendente.** A preencher depois da corrida real, com os números de `report.json`.

- Commit, rodada, janela e critério: _pendente_.
- Métricas por braço (status, regiões por frame, falhas de parse e de chamada, abstenção por modo, entidades, relações, tempo e pico de RSS por passo): _pendente_.
- Braços de reuso (percepção consumida = percepção do baseline): _pendente_.
- Concordância baseline × réplica: _pendente_.
- Leitura do mantenedor no viewer: _pendente_.

## Dry run na CI

`tests/end_to_end/test_perception_ablation_campaign_harness.py` roda este mesmo comando sobre a fundação sintética da CI, com uma janela sintética da sequência dela (três imagens, uma por segundo, das quais o passo 6 seleciona a primeira) e o PTv3 pelo dublê determinístico do `PTv3Runtime`. Todos os estágios, menos a percepção, são compostos pela runtime a partir da configuração; a percepção é o dublê de `tests/end_to_end/chain.py`, carimbado com a hora de cada execução e com uma feature densa DINOv3 por frame extraída pelo adapter real sobre uma saída fixa (`--dry-run-executors chain:campaign_dry_run_executors`), e o CLIP de texto do scoring é um dublê declarado em `resources.providers`. Uma camada comum põe nos braços a política de request semântico do baseline e o `checkpoint` e a `revision` do CLIP de `region_features`, que a base da CI não traz. O teste lê `maps.json` sem parar enquanto a campanha roda e confere que toda leitura é um índice válido e que ele cresce braço a braço; confere os 14 braços, o reuso da percepção do baseline pelo mesmo content hash, as camadas voxel da geometria da fundação e o registro da janela e dos parâmetros. Confere também uma falha no meio (sem a feature densa, o braço do canal denso falha, e o índice fica com os dois braços anteriores, válido) e as recusas: janela de outra sequência, sem imagem, noutro clock, com outro schema, fora dos limiares (medidos ou não), rodada existente, placeholder, diferença não declarada, parâmetro diferente do registrado, scorer fora do espaço das features de região (outro `checkpoint` ou outra `revision`), árvore suja; e uma falha do Eagle 2.5, o último braço, que deixa os outros 13 no índice e a concordância da réplica no relatório; a aceitação explícita das lacunas e o run real de um checkout limpo rodam de uma cópia do checkout (`checkout_copy`). Nada disso é evidência sobre dados reais.

## Limitações declaradas

- Sem reference set do corridor-02 (#581), nenhuma métrica é afirmação de qualidade.
- A réplica é um único par baseline × réplica: mede o piso de ruído desse par, não a variância.
- O braço do canal denso não tem `ContextRun` nem `ContextBuild`: a runtime não oferece `context run` com a percepção fornecida. `contextmap context run --select visual_perception=... --catalog ...` sai com status 0 e **ignora** a seleção, reexecutando a percepção; por isso o harness nunca o usa (achado da runtime, fora do escopo desta issue).
- O scoring não muda o mapa: a pontuação é um canal de evidência que nenhuma política de fusão consome (#180).
- O braço Florence-2 não tem claims de cena (ver o item 6 acima).

## Anexo: protocolo registrado (cópia literal)

Seção "Protocolo (registrado antes da execução)" da issue #670, de 2026-09-27:

**Dados**

- Trecho de 2:30 a 2:35 do `corridor-02.bag`, a partir da primeira mensagem do bag, ingerido pela issue #669.
- Seleção: uma imagem a cada 6 frames da câmera, cerca de 20 imagens.
- Uma única `SpatialFoundation` (sequência, trajetória ground truth pela ponte de pose, mapa geométrico bruto) para todos os braços.

**Baseline**

| Componente | Configuração |
|---|---|
| Discovery | SAM2 automático |
| Features densas | DINOv3 |
| Features de região | CLIP |
| Interpretação semântica | Qwen3-VL-4B (4bit, `temperature` 0), `scene/v1` + `region/v1` |
| View | `tight_crop` |
| Fusão | baseline |
| Associação | sem canal denso |
| Point representation | PTv3 ligado (opt-in, #663) |

**Braços.** Cada um varia um eixo a partir do baseline e tem a sua branch de contexto e o seu `ContextMap`:

| Grupo | Braços | Execução |
|---|---|---|
| Discovery × prompt de região | SAM2 e SAM3 (conjunto de conceitos, #662) × `region/v1`, `region-abstention/v1`, `region-scene-context/v1`, mais a réplica do baseline | percepção completa |
| Intérprete | Eagle 2.5; Florence-2 semântico | percepção completa |
| Views do Qwen | `contextual_crop`; `masked_subject` | percepção completa |
| Fusão ciente de qualidade | `quality-aware-evidence-accumulation-v1` | build extra sobre a branch do baseline, com a percepção e a associação dele; não reinfere nada |
| Canal denso DINOv3 na associação | `dense-feature-sampling-v1` | reusa exatamente o artifact de percepção do baseline (mesmo content hash); só a associação e o que vem depois são refeitos |
| Scoring semântico CLIP | estágio `semantic_scoring` | sobre o run de percepção do baseline |

**Camadas voxel** (derivadas só da geometria, iguais para todos os braços): a parte geométrica do harness #624 sobre o mapa da foundation, em 0,05 e 0,10 m.

**Parâmetros dos novos eixos.** Ficam na configuração da campanha, registrados em `protocol.json` com a origem de cada um. O mantenedor pode trocá-los antes da execução.

| Parâmetro | Valor | Origem |
|---|---|---|
| `context_margin_ratio` | 0,5 | exemplo de `runtime/docs/composition.md` |
| `mask_fill_rgb` | `[0, 0, 0]` | exemplo de `runtime/docs/composition.md` |
| Rampas da fusão ciente de qualidade | profundidade (bom 4 m, ruim 30 m) e distância à borda (bom 60 px, ruim 10 px), fator neutro 0,5, `observation-quality-v3` | braço `quality_aware_depth_border` do experimento `semantic-fusion-corridor-02-20260921` |
| Interpolação do canal denso | `bilinear` | escolha registrada |
| `max_dynamic_tiles` do Eagle 2.5 | 6 | escolha registrada |
| Task do Florence-2 semântico | `<REGION_TO_CATEGORY>` | escolha registrada |
| Template do scoring CLIP | `a photo of a {hypothesis}.` | escolha registrada |

**Fora da campanha**

- Debug full (decisão do mantenedor).
- AlphaCLIP (#71).
- Ablação de PTv3: ele é um controle fixo.
- Voxel como estágio da runtime (#664).

**Métricas.** As do piloto (#665), sem anotação, por braço, mais a concordância baseline × réplica. Para os braços de reuso, também a identidade dos artifacts reusados, que prova que nada foi reinferido.

### Reuso com produtores independentes

O harness atual grava relatório e índice no schema `v2`. Braços de reuso selecionam explicitamente os artifacts regional, visual e semântico do baseline. O braço de fusão reutiliza também a associação; o braço denso reexecuta associação usando regiões e features visuais selecionadas, enquanto a fusão lê a interpretação semântica selecionada. O braço de scoring lê os três artifacts diretamente. Cada reuso confere contrato, identidade e hash de conteúdo de cada produtor consumido e exige disposition `provided`; uma divergência encerra a campanha. Os catálogos registram a sequência e a dependência regional dos produtores visual/semântico, e `maps.json` publica os três caminhos separados. Bundles históricos de cenários e configurações permanecem congelados.
