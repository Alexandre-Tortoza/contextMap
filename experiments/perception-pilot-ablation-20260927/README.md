# Piloto da ablação de percepção, um mapa por braço (issue #665)

Antes das runs completas da campanha de percepção, este piloto confere que cada braço da matriz roda de ponta a ponta sobre dados reais e publica o próprio `ContextMap`, comparável aos outros no viewer. Cada braço tem a sua branch de contexto e, portanto, o seu mapa; todos partem da mesma `SpatialFoundation` (sequência, trajetória e mapa geométrico), que mantém a geometria como controle. O protocolo foi registrado na #665 antes de qualquer corrida e está copiado literalmente no [anexo](#anexo-protocolo-registrado-cópia-literal).

> **Estado: preparado, ainda não executado em dado real.** O harness, as camadas de configuração, as verificações e as métricas estão implementados e testados; um dry run sobre a cadeia de CI percorre os sete braços pela CLI na suíte (`tests/end_to_end/test_perception_pilot_harness.py`). Este ambiente não tem GPU, checkpoints nem o corridor-02: não há número real a citar até o comando abaixo rodar na máquina com GPU e o resultado ser registrado na seção [Resultado](#resultado).

## Braços

| Braço (arquivo em `config/arms/`) | Region discovery | Prompt de região | Papel |
|---|---|---|---|
| `sam2-region-v1` | SAM2 automático | `region/v1` | baseline |
| `sam2-region-abstention-v1` | SAM2 automático | `region-abstention/v1` | variante |
| `sam2-region-scene-context-v1` | SAM2 automático | `region-scene-context/v1` (com `region_scene_context: true`) | variante |
| `sam3-region-v1` | SAM3 com conjunto de conceitos | `region/v1` | variante |
| `sam3-region-abstention-v1` | SAM3 com conjunto de conceitos | `region-abstention/v1` | variante |
| `sam3-region-scene-context-v1` | SAM3 com conjunto de conceitos | `region-scene-context/v1` (com `region_scene_context: true`) | variante |
| `sam2-region-v1-replica` | SAM2 automático | `region/v1` | réplica do baseline, mede o ruído |

Os braços rodam nessa ordem: o baseline primeiro, a réplica por último.

## Como o harness implementa o protocolo

| Protocolo | Implementação |
|---|---|
| Uma `SpatialFoundation` para todos os braços | `--foundation-run`: um run concluído cuja execução nomeia sequência, trajetória e mapa, o mesmo que `contextmap context branch create --from-run` aceita. As sete branches nascem dele |
| Um braço = uma branch = um mapa | branch `pilot-run-NNNN-<braço>`, com uma `ContextRun` (revisão 1) e um `ContextBuild` que materializa o `ContextMap` |
| Baseline e variáveis | [`config/base.template.json`](config/base.template.json) é o baseline; cada braço acrescenta a própria camada, [`config/arms/<braço>.json`](config/arms/), que declara só as duas variáveis: o backend de region discovery (e, nos braços SAM3, o bloco `sam3` e o provider dele) e `qwen.prompt_policy.region` (e `region_scene_context`) |
| PTv3 ligado em todos os braços | [`config/point-representation.template.json`](config/point-representation.template.json), passada por `--all-arms-config`, entra em todo braço depois da base e antes da camada do braço |
| Controles fixos | verificados antes de qualquer execução: cada par de braços só difere onde uma variável que eles atribuem de forma diferente permite (`contextmap.evaluation.arm_differences`, campo a campo, sobre a configuração resolvida) |
| Seleção congelada | `--selection-file`, por padrão o snapshot do cenário 1.0.5, [`solution-1-canonical-1.0.5-corridor-02-sample.json`](../../src/contextmap/evaluation/docs/scenarios/solution-1-canonical-1.0.5-corridor-02-sample.json) (a issue cita `docs/scenarios/`; o arquivo está em `src/contextmap/evaluation/docs/scenarios/`). As 20 imagens entram em todo `context run` como `inputs.observation_selection` do tipo `explicit_ids` |
| `context run` e `context build` em processo novo | por braço, três comandos da CLI instalada, cada um num processo novo, medido por `measure_child_process` e executado pelos passos do #501 (abaixo) |
| Métricas sem anotação | lidas dos artifacts publicados pelos leitores públicos; ver [Métricas](#métricas) |
| Concordância baseline × réplica | `RegionJoinPolicy(policy_id="pilot-replica-join/v1", min_iou=0.5)`, declarada em `protocol.json` |
| Critério | `report.json`, `criterion`: sete braços completos, sete mapas verificados, sete `context_map_id` distintos |

Onde o protocolo não fixa o detalhe, o harness decide assim, e cada escolha fica registrada em `protocol.json`:

1. **Nome das branches.** `pilot-run-NNNN-<braço>`, não só `<braço>`. Uma rodada que falha (fail-fast) é refeita com `--run-number` seguinte no mesmo workspace, sobre a mesma fundação; com o nome do braço sozinho, a segunda rodada encontraria as branches da primeira. É a convenção do #501 (`partition-run-NNNN-whole`).
2. **Camadas.** A configuração de um braço é, nesta ordem de precedência da runtime: a base, as camadas comuns (`--all-arms-config`, na ordem dada) e a camada do braço. O bloco `sam3` é repetido nos três braços SAM3; a verificação par a par garante que os três ficaram iguais.
3. **O que a verificação compara.** Cada componente (backend e parâmetros, como a runtime os guarda), `pipeline`, `inputs` e `policies`. `resources` fica de fora: é implantação (qual processo carrega qual modelo, em qual device, em qual workspace), e um braço SAM3 declara outro provider de region discovery que um braço SAM2. Uma diferença não declarada, ou um braço cuja configuração não mostra o nível que ele diz atribuir, recusa a rodada antes de criar qualquer coisa, com o campo e os dois valores.
4. **Placeholders.** Qualquer valor da configuração resolvida de um braço que ainda contenha `REPLACE` recusa a rodada antes de criar qualquer coisa. Sem isso, um placeholder de texto passaria pela composição (a versão de modelo do SAM só precisa não ser vazia) e iria para a proveniência como se fosse o valor da máquina.
5. **Seleção fora da fundação.** Recusada antes de criar qualquer coisa quando uma imagem do snapshot não está entre as imagens da sequência da fundação, ou quando o snapshot fixa outra sequência (`subject.sequence_artifact_id`; a amostra do corridor-02 foi congelada sobre `720a486de8d44c16a9d3d2ff9fa7b1a4`).
6. **A réplica é outra inferência.** A identidade de uma `ContextRun` vem do conteúdo dos artifacts, e duas execuções que referenciam a mesma evidência são a mesma `ContextRun`. A percepção real grava em cada resultado quando o produziu (`created_at`) e os tempos de cada estágio, então a réplica sempre publica outro conteúdo, outra `ContextRun` e outro mapa. O dublê do dry run faz o mesmo (`chain.replicable_dry_run_executors`).

## Métricas

Todas por braço, lidas pelos leitores públicos dos artifacts; nenhuma é combinada com outra, e nenhuma é afirmação de qualidade (não há reference set do corridor-02, #581).

| Métrica | Onde, em `report.json` | Definição |
|---|---|---|
| status e falhas | `arms.<braço>.status`, `failure` | `complete`, `failed` (o braço em que a rodada parou) ou `not_run`; a falha nomeia braço, passo, mensagem da CLI e dica |
| regiões por frame | `arms.<braço>.perception.regions_per_frame` | regiões aceitas de cada resultado do run de percepção: mínimo, mediana, máximo e total |
| requests semânticos | `...perception.semantic_requests` | `account_semantic_attempts`: `attempted = parsed + parse_failed + call_failed`, reconciliados por `request_id` nos três streams contratuais; `parse_failure_rate` sobre os respondidos e `call_failure_rate` sobre os emitidos, `null` quando o denominador é zero |
| falhas de chamada de backend | `...perception.backend_call_failures` | as chamadas de `outputs/semantic-backend-call-failures.jsonl`, lidas por `iter_failed_semantic_backend_calls`, contadas por `error_type` |
| abstenção | `...perception.abstention.<modo>` | por modo (`scene`, `region`), respostas parseadas que se abstiveram sobre as respostas parseadas |
| labels | `...perception.region_labels` | o label de uma região é o conjunto das hipóteses das claims primárias dela, texto exato; regiões rotuladas, labels distintos e os 20 mais frequentes (empate pela ordem alfabética) |
| entidades e relações | `arms.<braço>.build.counts` | entidades resolvidas e relações do `ContextMap` (por estado), com as contagens da fusão e do semantic mapping, lidas por `read_build_state` (a leitura do #501) |
| artifacts verificados | `arms.<braço>.build.validation`, `integrity_problems` | validação `FULL` do `ContextMap` e integridade dos runs lidos |
| tempo e memória | `arms.<braço>.steps.<passo>` | tempo de parede e pico de RSS do processo de cada comando (`branch`, `context_run`, `build`) |
| concordância baseline × réplica | `replica_agreement` | as regiões aceitas do baseline ficam do lado de referência de `join_frame_regions` (um-para-um, guloso por IoU de máscara, `min_iou` 0,5); só os pares rotulados dos dois lados são comparados: `agreement = agreeing / compared`, `null` quando nada foi comparado. Pares rotulados de um lado só, pares sem label e regiões sem par ficam contados à parte |

## Pré-requisitos na máquina com GPU

**Gates da issue.** Só rode o comando com:

- a #661 (providers pela CLI) na `dev`: sem ela, nenhum braço com SAM2, SAM3 ou Qwen roda pela CLI;
- a #663 na `dev`, para o PTv3: sem ela o estágio `point_representation` não tem executor, e o primeiro `context run` para no preflight;
- a #662, e a #610 medida, antes dos braços SAM3: sem a #662, `Sam3Config` não aceita `prompts` e os braços SAM3 falham na composição, depois de os braços SAM2 já terem rodado (a rodada é fail-fast, então para ali).

**Ambiente.** Checkout limpo no commit que será citado, com `pip install -e '.[vision]'` (DINOv3 e CLIP usam o loader Hugging Face empacotado) e o ambiente do Pointcept que o provider do PTv3 exige ([`ptv3.md`](../../src/contextmap/point_representation/docs/ptv3.md)). O harness confere isso: um run real é recusado com status 2, antes de criar qualquer coisa, quando nenhum commit identifica o checkout, quando a árvore não está comprovadamente limpa (mudança ou arquivo não rastreado fora do `.gitignore`, ou `git status` indisponível) ou quando `--code-identity` não nomeia exatamente o commit do checkout, porque os braços rodariam um código que a identidade registrada não identifica. A mensagem lista cada motivo: faça commit ou stash e rode de novo num checkout limpo, e passe o commit do checkout em `--code-identity` ou omita a opção; a cópia preenchida da configuração fica em `outputs/`, que o `.gitignore` já exclui. A regra é a mesma dos harnesses da #501 e da #504 (`resolve_code_identity`, no [`steps.py`](../incremental-partition-invariance-20260927/scripts/steps.py) da #501); o dry run é isento.

**Providers.** SAM2, SAM3, Qwen e PTv3 não têm loader no repositório: cada um precisa de um `RuntimeProvider` (`Callable[[config, ResolvedSecrets], runtime]`) num módulo importável da máquina, declarado em `resources.providers` ([`composition.md`](../../src/contextmap/runtime/docs/composition.md#providers-declarados-em-configuração-resourcesproviders)). A base declara os de SAM2 (`visual_perception.region_discovery`) e Qwen (`visual_perception.semantic_interpretation`); cada camada SAM3 troca o de region discovery pelo do SAM3; a camada do PTv3 declara `point_representation.encoder`. Sem providers o primeiro `context run` para no preflight, antes de qualquer `ContextRun`, e o relatório sai `failed` com a causa e a dica.

**Fundação.** Um run concluído no workspace cuja execução nomeia a sequência, a trajetória e o mapa, construído sobre a sequência ingerida do corridor-02 que a amostra congelada fixa (`720a486de8d44c16a9d3d2ff9fa7b1a4`). É o que `contextmap context branch create --from-run` aceita; ver "Fundação" no [README do #501](../incremental-partition-invariance-20260927/README.md#pré-requisitos-na-máquina-com-gpu). O harness não cria a fundação.

**Configuração.** Copie `config/` para fora do Git e preencha cada placeholder da cópia; o harness recusa a rodada enquanto restar um.

| Placeholder | Arquivo | O que o substitui |
|---|---|---|
| `REPLACE-WITH-THE-SEQUENCE-ARTIFACT-ID` | base | a identidade do `SequenceArtifact` da fundação (`720a486de8d44c16a9d3d2ff9fa7b1a4`) |
| `REPLACE_WITH_YOUR_LOADERS_MODULE:load_sam2`, `:load_qwen` | base | os alvos `módulo:atributo` dos providers de SAM2 e Qwen |
| `REPLACE-WITH-THE-SAM2-MODEL-VERSION` | base | a versão do checkpoint `facebook/sam2.1-hiera-tiny` que o provider carrega |
| `REPLACE-WITH-THE-40-HEX-DINOV3-REVISION` | base | o SHA completo do commit de `facebook/dinov3-vitb16-pretrain-lvd1689m`; o validado na #69 é `5931719e67bbdb9737e363e781fb0c67687896bc` ([`dinov3-validation.md`](../../src/contextmap/visual_perception/docs/dinov3-validation.md)) |
| `REPLACE-WITH-THE-40-HEX-CLIP-REVISION` | base | o SHA completo do commit de `openai/clip-vit-large-patch14` |
| `REPLACE-WITH-THE-40-HEX-QWEN-REVISION` | base | o SHA completo do commit de `Qwen/Qwen3-VL-4B-Instruct` |
| `REPLACE-WITH-THE-QWEN-PRECISION` | base | `float32`, `float16` ou `bfloat16`, o dtype de cômputo da quantização em 4 bits |
| `REPLACE-WITH-THE-QWEN-TOKEN-BUDGET` | base | o `max_new_tokens`, um inteiro positivo |
| `REPLACE_WITH_YOUR_LOADERS_MODULE:load_sam3` | as três camadas SAM3 | o alvo do provider do SAM3 |
| `REPLACE-WITH-THE-SAM3-MODEL-VERSION` | as três camadas SAM3 | a versão do checkpoint `facebook/sam3` que o provider carrega |
| `REPLACE-WITH-THE-CONCEPT-SET` | as três camadas SAM3 | a lista ordenada de frases do conjunto de conceitos (#662), **a mesma nos três braços** (a verificação par a par recusa conjuntos diferentes) |
| `REPLACE_WITH_YOUR_LOADERS_MODULE:load_ptv3` | point representation | o alvo do provider do PTv3 (`PointceptPTv3Runtime`, exemplo em [`ptv3.md`](../../src/contextmap/point_representation/docs/ptv3.md)) |

Os valores que o repositório registra já estão preenchidos: o checkpoint do SAM2 e a precisão `float32` de SAM2, DINOv3 e CLIP ([modelo da v0.2.0](../../examples/v0.2.0/README.md)); o modelo, a quantização `4bit` e `temperature` 0 do Qwen, a `prompt_policy` e a `view_policy` `tight_crop` (protocolo); o SAM3 com `facebook/sam3`, `bfloat16` e a estratégia `text_prompt` (#662); o PTv3 medido na RTX 3060 (`ptv3m1-base`, checkpoint nuScenes com o hash fixado, `float32`, grade de 0,05 m, pooling `center`, sem normalização, `padding_channels` 1, `min_support_points` 3, suporte por raio de 0,5 m com centralização, [`ptv3.md`](../../src/contextmap/point_representation/docs/ptv3.md)) e a seleção de centros `observation-support-anchor-v1`, uma âncora por observação espacial da associação do braço ([`center-selection.md`](../../src/contextmap/point_representation/docs/center-selection.md), #663); e as políticas de jusante, iguais às do template do #501 ([`corridor-02.template.json`](../incremental-partition-invariance-20260927/config/corridor-02.template.json)). Um parâmetro que o template não escreve fica com o default da classe de configuração do backend.

**Custo estimado.** ~37 s/frame na RTX 3060 para a percepção do v0.1.0 (SAM2, DINOv2, CLIP e Qwen): 7 braços × 20 imagens ≈ 1 h 30 min só de percepção, sem o SAM3 e o PTv3, que não têm custo medido neste recorte. O pico de RSS de `sensor_association` foi de 25 GB no v0.1.0 (README do #501).

## Comando

Da raiz do checkout, depois de copiar e preencher a configuração:

```bash
cp -r experiments/perception-pilot-ablation-20260927/config outputs/perception-pilot-config
# preencher cada REPLACE... de outputs/perception-pilot-config/ (tabela acima)

python experiments/perception-pilot-ablation-20260927/scripts/run_pilot.py \
    --workspace outputs/<workspace> \
    --foundation-run outputs/<workspace>/<dataset>/run-NNNN \
    --base-config outputs/perception-pilot-config/base.template.json \
    --arms-dir outputs/perception-pilot-config/arms \
    --all-arms-config outputs/perception-pilot-config/point-representation.template.json
```

Padrões: `--selection-file` é a amostra congelada do cenário 1.0.5; `--code-identity` é o `HEAD` do checkout, o mesmo em todo `context run` e `context build` (um controle fixo do protocolo); `--run-number 1` numera a rodada. Num run real, um `--code-identity` dado tem de nomear exatamente o commit do checkout (um SHA abreviado dele é aceito), e o que `protocol.json` e cada comando gravam é sempre o SHA completo desse commit; no dry run o valor dado é gravado como veio.

Por braço, o harness executa três comandos da CLI instalada, cada um num processo novo:

```bash
contextmap context branch create pilot-run-NNNN-<braço> --from-run <fundação> -c <base> -c <comuns...> -c <braço> --workspace <ws> --json
contextmap context run --branch pilot-run-NNNN-<braço> --code-identity <id> -c <base> -c <comuns...> -c <braço> --set 'inputs.observation_selection={"kind": "explicit_ids", ...}' --workspace <ws> --json
contextmap context build --branch pilot-run-NNNN-<braço> --revision 1 --code-identity <id> -c <base> -c <comuns...> -c <braço> --set 'resources.providers={}' --workspace <ws> --json
```

Os comandos passam por `scripts/context_step.py` do #501, que chama `contextmap.runtime.cli.main` no próprio processo, sem injetar nada, só para gravar a resposta JSON num arquivo; `measure_child_process` mede o tempo de parede e o pico de RSS de cada um. O build roda sem providers porque a CLI compõe todo estágio que a configuração permite e, com eles, carregaria os modelos de percepção que um build não usa.

**Reuso dos passos do #501.** `run_pilot.py` importa `steps.py` de [`../incremental-partition-invariance-20260927/scripts/`](../incremental-partition-invariance-20260927/scripts/steps.py) pondo aquele diretório no caminho de import, como o #504 já faz: o comando medido, a leitura da resposta, `StepFailed`, o registro do passo e do build, a configuração sem caminho da máquina e a regra da identidade do código são os mesmos, e os três harnesses continuam cobertos pelos próprios dry runs. Os módulos deste diretório só têm o que é do piloto: a matriz de braços e as verificações das camadas (`pilot_arms.py`), a seleção congelada (`pilot_selection.py`) e as métricas (`pilot_metrics.py`).

A execução é **fail-fast**: o primeiro comando que falha encerra a rodada; `report.json` sai com `"status": "failed"`, o braço e o passo, e os braços seguintes ficam `not_run`. Tudo que pode recusar a rodada sem rodar um modelo é conferido antes de criar o diretório de saída (status 2): seleção fora da fundação, placeholder, diferença não declarada entre braços, um diretório de rodada que já existe, que nunca é sobrescrito, e, num run real, árvore suja, commit não identificado ou um `--code-identity` que não é o commit do checkout (ver "Ambiente" em [Pré-requisitos](#pré-requisitos-na-máquina-com-gpu)). O status de saída é 0 só quando o critério passa.

## O que o comando grava

Em `outputs/perception-pilot-ablation-20260927/run-NNNN/harness/` (fora do Git):

| Arquivo | Conteúdo |
|---|---|
| `protocol.json` | gravado **antes** do primeiro comando: commit e estado da árvore, identidade do código (num run real, o SHA completo do commit), a fundação, a seleção (snapshot, SHA-256, ids), as variáveis, e por braço o papel, a branch, as atribuições, a configuração efetiva (documento e digest, sem o caminho do workspace), a configuração de cada componente de percepção, os alvos de provider e as diferenças em relação ao baseline; a política de join da réplica e a definição dos labels |
| `report.json` | por braço: status, passos (status de saída, tempo, pico de RSS, run relativo ao workspace, disposição de cada estágio, bytes), métricas de percepção e o registro do build (identidades, contagens, validação, integridade, bytes por artifact); a concordância baseline × réplica; o critério; a falha, quando houver |
| `maps.json` | por braço: branch, `context_map_id` e os caminhos, relativos ao workspace, do diretório do `ContextMap`, do run de percepção e do run de representação de pontos (`null` sem PTv3), para o viewer |
| `steps/<braço>/` | o pedido e a resposta JSON de cada comando e, no passo que falhou, o fim do stderr. Têm caminhos da máquina e não vão para o Git |

Os artifacts ficam no workspace, onde a runtime os publica; o relatório e `maps.json` os citam por caminho relativo a ele. `protocol.json`, `report.json` e `maps.json` não gravam nenhum caminho absoluto da máquina: a mensagem de uma falha troca o workspace, a raiz de saída, o checkout e o home por `<workspace>`, `<outputs>`, `<checkout>` e `~`.

## O que volta para o Git

Depois da corrida real, numa árvore limpa, no commit citado:

1. Copiar para este diretório `protocol.json`, `report.json` e `maps.json` (não os `steps/`, nem os artifacts do workspace).
2. Escrever o `manifest.json` do bundle com o SHA-256 completo de cada arquivo executado (`scripts/`, `config/`) e de cada registro copiado, como pede [`docs/ARTIFACTS.md`](../../docs/ARTIFACTS.md); a partir daí `tests/evaluation/test_experiment_bundles.py` confere o bundle. As cópias preenchidas da configuração não voltam: a configuração efetiva de cada braço já está em `protocol.json`, sem caminho da máquina.
3. Preencher a seção [Resultado](#resultado).

## Resultado

> **Pendente.** A preencher depois da corrida real, com os números de `report.json`.

Regra de decisão, a do protocolo: o piloto **passa** quando os 7 braços terminam com artifacts verificados e mapas publicados (`criterion.passes` em `report.json`). A escolha de quais braços seguem para as runs completas é do mantenedor, com base no viewer e nas métricas, e fica registrada aqui com a justificativa.

- Commit, rodada e critério: _pendente_.
- Métricas por braço (status, regiões por frame, taxa de falha de parse, falhas de chamada, abstenção por modo, entidades, relações, tempo e pico de RSS por passo): _pendente_.
- Concordância baseline × réplica (`pilot-replica-join/v1`, `min_iou` 0,5): _pendente_.
- Braços escolhidos para as runs completas e justificativa: _pendente_.

## Dry run na CI

`tests/end_to_end/test_perception_pilot_harness.py` roda este mesmo comando sobre a fundação sintética da CI: todos os estágios, menos a percepção, são compostos pela runtime a partir de uma configuração parametrizada, com as camadas de braço deste diretório (placeholders preenchidos) por cima, e a percepção é o dublê determinístico de `tests/end_to_end/chain.py`, carimbado com a hora de cada execução (`--dry-run-executors chain:replicable_dry_run_executors`, que marca o relatório como `dry_run`). A seleção é o snapshot do subconjunto sintético do cenário 1.0.5 (3 imagens), e a camada do PTv3 entra num teste próprio, com o dublê determinístico do `PTv3Runtime` como provider (`pointrep_ptv3_fakes:provide_fake_ptv3`). O dry run é permitido numa árvore suja, como a da CI e a do desenvolvimento, e só registra o estado dela em `protocol.json` (`code.dirty`). O teste confere os sete braços com a branch na revisão 1, sete `context_map_id` distintos, o relatório completo e a ausência de caminhos da máquina; e as recusas: providers ausentes (a rodada para no primeiro `context run`, sem nenhuma `ContextRun`), seleção fora da fundação, snapshot fixado em outra sequência, rodada existente, placeholder deixado e diferença não declarada. As recusas do run real por árvore suja, por falta de commit e por um `--code-identity` que nomeia outro commit, e o run real que as passa (que grava o SHA completo do commit, com `--code-identity` omitido ou abreviado), rodam o script de uma cópia do piloto e dos scripts do #501 num repositório git temporário (`checkout_copy`, em `tests/end_to_end/incremental_harness.py`, a mesma dos testes do #501 e do #504), que é o checkout que o script registra; assim nenhum teste depende do estado da árvore de quem os roda. Os testes das camadas conferem que cada uma resolve pela runtime, difere do baseline só na variável declarada e resolve a política de request semântico do seu prompt; que a base declara o baseline do protocolo e compõe todo estágio menos a percepção; e que a percepção de cada braço compõe, com os valores da máquina preenchidos e runtimes falsos no lugar dos modelos, os três braços SAM3 inclusive (#662). Com a camada do PTv3, cada braço publica o seu run de representação de pontos, localizado em `maps.json`. Nada disso é evidência sobre dados reais.

## Limitações declaradas

- Sem reference set do corridor-02 (#581), nenhuma métrica é afirmação de qualidade: são contagens, taxas e concordância, lidas lado a lado.
- A réplica é um único par baseline × réplica: mede o piso de ruído desse par, não a variância dele entre execuções.
- A concordância usa o texto exato das hipóteses; sinônimos contam como discordância.

## Anexo: protocolo registrado (cópia literal)

Seção "Protocolo (registrado antes da execução)" da issue #665, de 2026-09-27:

**Hipótese (operacional, não de qualidade).** Todos os braços da matriz rodam sobre o recorte real e publicam, cada um, um `ContextMap` auditável. As diferenças entre braços aparecem nas métricas sem anotação e são maiores que o ruído medido pela réplica do baseline. Sem o #581 não há afirmação de qualidade contra referência.

**Baseline.**

| Componente | Configuração |
|---|---|
| Discovery | SAM2 automático |
| Features densas | DINOv3 |
| Features de região | CLIP |
| Interpretação semântica | Qwen3-VL-4B (4bit, `temperature` 0) |
| Prompts | `scene/v1` + `region/v1` |
| View | `tight_crop` |
| Point representation | PTv3 ligado (opt-in, #663) |

**Variáveis (fatorial completo 2×3 = 6 braços):**

- region discovery ∈ {SAM2 automático, SAM3 com conjunto de conceitos (#662)};
- prompt semântico de região ∈ {`region/v1`, `region-abstention/v1`, `region-scene-context/v1`}.

**Réplica.** Uma réplica do baseline, em branch própria, mede o piso de ruído do Qwen (#556). Total: 7 branches, 7 mapas.

**Controles fixos.**

- DINOv3 é o único backend denso. O DINOv2 fica fora por decisão do mantenedor: o DINOv3 é a melhoria.
- CLIP, parâmetros do Qwen, view policy, todas as políticas downstream, foundation, `code_identity` e seleção.

**Seleção.** A amostra congelada de 20 imagens do cenário 1.0.5 do corridor-02 (`docs/scenarios/*1.0.5*corridor-02-sample.json`).

**Métricas por braço, sem anotação:**

- status e falhas;
- regiões por frame;
- requests semânticos, taxa de falha de parse, falhas de chamada de backend (`semantic-backend-call-failures.jsonl`), taxa de abstenção e histograma de labels;
- entidades e relações do `ContextMap`;
- tempo de parede e pico de RSS por passo.

Entre baseline e réplica, a concordância de label sobre regiões pareadas por IoU, com a política de join declarada (`RegionJoinPolicy` com id e `min_iou` explícitos).

**Critério.** O piloto passa quando os 7 braços terminam com artifacts verificados e mapas publicados. A escolha de quais braços seguem para runs completas é do mantenedor, com base no viewer e nas métricas, e fica registrada na seção "Resultado" do README.

**Gates antes de rodar.**

- #661 (providers pela CLI) na `dev`;
- #662 e o #610 medido, antes dos braços SAM3;
- #663 na `dev`, para o PTv3.

### Leitura após a separação das capabilities

O harness atual publica report e índice de mapas no schema `v2`. Cada braço localiza explicitamente `region_artifact`, `visual_evidence_artifact` e `semantic_interpretation_artifact`; a métrica regional usa o artifact regional, e as tentativas/labels usam o artifact semântico que o selecionou. A composição de vistas verifica sequência, seleção regional, integridade e cobertura por observação antes do IoU. As máscaras continuam pertencendo ao produtor regional. Os snapshots de cenário `1.0.5` e os resultados históricos permanecem congelados.
