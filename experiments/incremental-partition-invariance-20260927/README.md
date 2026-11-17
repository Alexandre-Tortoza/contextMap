# Invariância à partição do contexto incremental, braço real (issue #501)

Este experimento mede, sobre dados reais do corridor-02, se dividir as mesmas observações físicas em várias `ContextRun`s muda o que um `ContextBuild` conclui. O protocolo foi pré-registrado na #501 em 2026-09-26, antes de qualquer corrida, e está copiado literalmente no [anexo](#anexo-protocolo-pré-registrado-cópia-literal). O braço de CI, com percepção determinística, já roda em `tests/end_to_end/test_runtime_chain.py` (`TestPartitionInvariance`).

> **Estado: preparado, ainda não executado em dado real.** O harness, as métricas, o bootstrap e o relatório estão implementados e testados, e um dry run sobre a cadeia de CI percorre o caminho inteiro na suíte (`tests/end_to_end/test_partition_invariance_harness.py`). Este ambiente não tem GPU, checkpoints nem o corridor-02: não há número real a citar até o comando abaixo rodar na máquina com GPU e o resultado ser registrado aqui.

## O que o comando decide

| Critério | Camada | Resultado no `report.json` |
|---|---|---|
| **H1**: o estado da camada 1 de `C` é igual ao de `W1` e ao de `W2`, com a pré-condição `W1 == W2` | exata | `layer_one.outcome`: `holds`, `partition_divergence` ou `precondition_failed`, com cada dimensão comparada em `precondition`, `chunked_w1` e `chunked_w2` |
| **H2**: a partição não reduz a concordância semântica além do ruído de rerun, por métrica | estatística | `layer_two.<M_claim\|M_state\|M_label>.decision`: `rejects_invariance`, `equivalent_within_delta`, `inconclusive` ou `not_measurable`, com `effect` (ponto, IC, seed, reamostragens) e as concordâncias de cada par |
| **Cenário 2**: `R = [W1, W2]` mantém as observações físicas e soma os resultados de inferência | exata | `scenario_2.holds` |
| **Cenário 3**: uma política de jusante alterada reaproveita a evidência congelada | estrutural | `scenario_3.holds`, com os estágios de contexto recomputados (nenhum, se passa) e `outputs_differ` |

As métricas nunca são combinadas: cada contagem, cada métrica e cada decisão aparece em separado. As definições estão em [`incremental-context.md`](../../src/contextmap/evaluation/docs/incremental-context.md) e o código em `contextmap.evaluation.incremental_context`.

## Como o harness implementa o protocolo

| Protocolo | Implementação |
|---|---|
| Uma única `SpatialFoundation` | `--foundation-run`: um run concluído cuja execução nomeia sequência, trajetória e mapa, o mesmo que `contextmap context branch create --from-run` aceita. As duas branches do experimento nascem dele |
| `W1`, `W2`: uma `ContextRun` cada sobre os 100 frames | branch `partition-run-NNNN-whole`: `W1` e depois `W2` |
| `C`: 4 `ContextRun`s de 25 frames | branch `partition-run-NNNN-chunked`: `C1` a `C4`, consecutivos, que particionam exatamente as imagens de `W1` |
| Builds `W1`, `W2`, `C`, `R = [W1, W2]` e o do cenário 3 | `W1` = revisão 1 de `whole`; `W2` = `--member W2` da revisão 2; `R` = revisão 2; `C` = revisão 4 de `chunked`; `C_policy` = revisão 4 com a política de jusante alterada |
| Camada 1 | `read_build_state` normaliza cada build pelos seus artifacts públicos; `judge_layer_one` decide H1. Nada de ponto flutuante entra na camada 1, então não há tolerância: só igualdade |
| Camada 2 | `claim_agreement` (clusters = frames), `paired_agreement` (clusters = entidades), `partition_effect` (bootstrap pareado por cluster), `decide_equivalence` (δ) |
| 10.000 reamostragens, seed no manifesto, δ = 0,10 | padrões do comando (`--resamples`, `--seed 20260926`, `--delta`), gravados em `protocol.json` antes do primeiro estágio |

Onde o protocolo não fixa o detalhe, o harness decide assim, e cada escolha fica registrada em `protocol.json`:

1. **"100 frames" são 100 imagens consecutivas.** O custo de ~37 s/frame do v0.1.0 é por imagem. `s` (`--first-image`) é o índice da primeira imagem entre as imagens da sequência, na ordem canônica; o corridor-02 tem uma câmera.
2. **`W2` é outra execução de inferência sobre as mesmas imagens.** O `run_id` de uma percepção é função da configuração do estágio (seleção incluída) e das entradas, então `W1` e `W2` com a mesma seleção teriam a mesma identidade, e `R` não poderia fundi-las como duas inferências. `W1` e `C` usam `explicit_ids`; `W2` usa o `frame_range` do índice canônico que cobre exatamente as mesmas imagens. É a técnica do braço de CI (`[0, 9)` e `[0, 10)`). A variável que muda entre `W` e `C` continua sendo só a partição: `W1` e `C` têm o mesmo tipo de seleção.
3. **Cenário 3** altera `spatial_relations.candidate.proximity_radius_m` de 0,6 para 0,3 (`--scenario3-set`). O critério é estrutural, então qualquer política de jusante serve; esta só mexe nas relações.
4. **`M_state` e `M_label`** comparam as entidades da Semantic Mapping (`AmbiguityState`, o tipo nomeado no protocolo, e o conjunto de labels das hipóteses), pareadas pela chave geométrica. As entidades resolvidas do `ContextMap` entram na camada 1, pela geometria e pelos membros.
5. **Claims** são as `PerceptionResult.claims`, a mesma fonte do script dos 29/90 (`scripts-run5/15_visual_perception_content_equivalence.py`). O contexto de cena não entra.
6. **Clusters.** `M_claim`: todo frame que algum dos três builds percebeu; um frame sem claim entra com 0/0. `M_state`/`M_label`: toda entidade de qualquer dos três builds; uma entidade sem par entra com 0/0, porque a falta de par já é divergência da camada 1.
7. **Reamostragem sem nada a comparar** (possível com poucos clusters) fica fora do intervalo e é contada em `undefined_resamples`, nunca substituída.
8. **Intervalo percentil** de 95%, gerador `numpy.random.default_rng(seed)`.
9. **Precedência da regra.** As regras valem na ordem em que foram registradas: um IC inteiro abaixo de zero rejeita mesmo quando o limite inferior também fica acima de −δ.
10. **Cenário 2.** "Dobra os `inference_results`" é conferido como `R = W1 + W2`, que é `2·W1` quando as duas execuções contam o mesmo; os três números vão no relatório.

## Pré-requisitos na máquina com GPU

- Checkout limpo no commit que será citado, com `pip install -e '.[vision]'`. O extra `vision` (torch, transformers, Pillow) é exigido pelo preflight de DINOv2 e CLIP. O harness confere o checkout: um run real é recusado com status 2, antes de criar qualquer diretório, quando nenhum commit identifica o checkout, quando a árvore não está comprovadamente limpa (mudança ou arquivo não rastreado fora do `.gitignore`, ou `git status` indisponível) ou quando `--code-identity` não nomeia exatamente o commit do checkout, porque os passos rodariam um código que a identidade gravada não identifica. A mensagem lista cada motivo: faça commit ou stash e rode de novo num checkout limpo, e passe o commit do checkout em `--code-identity` ou omita a opção. A regra é uma só para os harnesses da #501, da #504 e da #665 (`resolve_code_identity`, em [`scripts/steps.py`](scripts/steps.py)); o dry run é isento.
- **Providers de SAM2 e Qwen.** Esses backends não têm loader no repositório: cada um precisa de um `RuntimeProvider` (`Callable[[config, ResolvedSecrets], runtime]`) num módulo importável da máquina, declarado em `resources.providers`, o mecanismo documentado em [`composition.md`](../../src/contextmap/runtime/docs/composition.md#providers-declarados-em-configuração-resourcesproviders). O provider devolve só o runtime do modelo carregado, `Sam2AutomaticMaskRuntime.from_model(model=..., config=config)` e `HuggingFaceQwenRuntime(config=config)`, sem `image_loader` nem `view_root`: os adapters de cada run resolvem as imagens preparadas e as views no diretório de rascunho do executor e entregam ao runtime os pixels de cada pass e os bytes verificados de cada view ([o que um provider devolve](../../src/contextmap/runtime/docs/composition.md#o-que-um-provider-devolve-661), #661). Nenhum script versionado mostra o carregamento dos modelos: os runs de percepção do v0.1.0 foram produzidos fora dos drivers de `experiments/e2e-real-canonical-run-20260923/`, que montavam os executores à mão. Use os mesmos checkpoints daquela corrida. Sem providers o primeiro `context run` para no preflight, antes de qualquer estágio, e o relatório sai `failed` com a causa.
- **Configuração.** [`config/corridor-02.template.json`](config/corridor-02.template.json) traz o perfil `canonical/1`, os backends do cenário 1.0.5 e os valores de política do run real do v0.1.0 (`scripts-run3/01..08`). Os campos `REPLACE-...` (checkpoints, revisões, precisão e orçamento de tokens do Qwen, identidade da sequência, módulo dos providers) não estão registrados no repositório e vêm da máquina e do run de percepção do v0.1.0. A política de request semântico do Qwen, que não tem padrão desde a ablação de percepção (#542, #524), não é um deles: o template declara a que reproduz o request do v0.1.0, `prompt_policy` `scene/v1`/`region/v1` e `view_policy` só com o recorte justo ([composição do runtime](../../src/contextmap/runtime/docs/composition.md#política-de-prompt-semântico-542)). `spatial_relations.geometric_predicate` e `contact_predicate` são opcionais no catálogo e ausentes do cenário 1.0.5, mas o run real do v0.1.0 os executou; o template os mantém.
- **Fundação.** Um run concluído no workspace cuja execução nomeia a sequência, a trajetória e o mapa, construído sobre a sequência inteira. É o que `contextmap context branch create --from-run` aceita; por exemplo `contextmap run --stage geometric_mapping -c corridor-02.json --workspace <ws> --select ingestion=<id> --catalog <catálogo>`, com o `SequenceArtifact` já ingerido listado no catálogo ([`selection.md`](../../src/contextmap/runtime/docs/selection.md#arquivo-de-catálogo)). O harness não cria a fundação.
- **Custo estimado** (protocolo): ~37 s/frame na RTX 3060, ~3 h para as 300 execuções de frame (`W1`, `W2` e `C`); o pico de RSS de `sensor_association` foi de 25 GB no v0.1.0.

## Comando

Da raiz do checkout:

```bash
python experiments/incremental-partition-invariance-20260927/scripts/run_real_arm.py \
    --workspace outputs/<workspace> \
    --foundation-run outputs/<workspace>/<dataset>/run-NNNN \
    -c corridor-02.json \
    --first-image <s>
```

Os padrões são os do protocolo: `--images 100 --chunks 4 --resamples 10000 --seed 20260926 --delta 0.10 --confidence 0.95`. `--code-identity` omitido é o `HEAD` do checkout; num run real, dado, tem de nomear exatamente o commit do checkout (um SHA abreviado dele é aceito), e o que `protocol.json` e cada passo gravam é sempre o SHA completo desse commit. No dry run o valor dado é gravado como veio. `--run-number` numera uma rodada nova.

Cada passo é **um comando** `contextmap context branch create`, `context run` ou `context build`, executado num processo novo por `scripts/context_step.py` e medido por `measure_child_process` (tempo de parede e pico de RSS do processo). A execução é **fail-fast**: o primeiro passo que falha encerra o run, e `report.json` sai com `"status": "failed"` e a causa. Só um `report.json` com `"status": "complete"` e status de saída 0 é evidência para a #501. Um diretório de saída que já existe é recusado (status 2): o relatório de outro run nunca é sobrescrito.

`context_step.py` chama `contextmap.runtime.cli.main` no próprio processo, sem injetar nada, exatamente como o binário instalado: a CLI compõe todos os estágios da configuração (a percepção por `resources.providers`) e grava o `--code-identity` que cada `context run` e `context build` recebe. O script existe só para gravar a resposta JSON num arquivo, a salvo do que uma biblioteca imprima na saída padrão, e para o dry run da CI, o único caso em que a percepção é substituída (`main(executors=...)`). Os builds rodam sem providers (`resources.providers={}`), porque a CLI compõe todo estágio que a configuração permite e, com eles, carregaria os modelos de percepção que um build não usa.

## O que o comando grava

Em `outputs/incremental-partition-invariance-20260927/run-NNNN/harness/` (fora do Git):

| Arquivo | Conteúdo |
|---|---|
| `protocol.json` | gravado **antes** do primeiro estágio: commit e estado da árvore, identidade do código (num run real, o SHA completo do commit), configuração efetiva (documento e digest, sem o caminho do workspace), a configuração de cada backend de percepção (os checkpoints), os alvos de provider, a fundação, `s` e as seleções exatas de `W1`, `W2` e de cada pedaço, o bootstrap (método, gerador, reamostragens, seed, confiança), δ e a mudança do cenário 3 |
| `report.json` | por passo: status de saída, tempo, pico de RSS, run (relativo ao workspace), disposição de cada estágio (`provided`, `executed`, `reused`, `recomputed`, de `execution.json`) e bytes gravados; por build: identidades, contagens (observações físicas, resultados de inferência, suportes, contribuições, excluídas, entidades, rejeitadas, resolvidas, relações por estado), validação `FULL` do `ContextMap`, integridade dos runs lidos, bytes por artifact e as contagens da fusão da camada 2; e os quatro critérios |
| `steps/` | o pedido e a resposta JSON de cada comando, para diagnóstico. Têm caminhos da máquina e não vão para o Git |

Os artifacts em si ficam no workspace, onde a runtime os publica; o relatório os cita por caminho relativo a ele.

## Regra de decisão

É a do protocolo, sem mudança:

- **H1.** Com `W1 == W2` na camada 1, qualquer `C ≠ W` é divergência causada pela partição e é tratada como bug até prova em contrário. Se `W1 ≠ W2`, a camada 1 não é exata sob rerun: é achado, e qualquer reclassificação é decisão nova.
- **H2**, por métrica, sobre `D = ½[agr(W1,C) + agr(W2,C)] − agr(W1,W2)`: IC inteiro abaixo de 0 **rejeita** a invariância (diferença científica real, a investigar antes de fechar a milestone); senão, limite inferior ≥ −δ é **equivalente dentro de δ**; senão, **inconclusivo** (limitação conhecida nas notas da v0.1.1).
- **Cenários 2 e 3**: exatos.

## Ao registrar o run real

1. Rodar o comando numa árvore limpa, no commit que será citado.
2. Copiar para este diretório `protocol.json` e `report.json` (não os `steps/`), e escrever o `manifest.json` do bundle com o SHA-256 completo de cada arquivo executado (`scripts/`, `config/`), como pede [`docs/ARTIFACTS.md`](../../docs/ARTIFACTS.md). A partir daí `tests/evaluation/test_experiment_bundles.py` confere o bundle.
3. Resumir aqui H1, as três métricas de H2 com IC e decisão, e os cenários 2 e 3; classificar cada diferença da camada 1 como pede a issue (identidade/proveniência, tolerância numérica, ordem/id local, diferença científica real, entrada divergente, não determinismo).

## Dry run na CI

`tests/end_to_end/test_partition_invariance_harness.py` roda este mesmo comando sobre a fundação sintética da CI: todos os estágios, menos a percepção, são compostos pela runtime a partir de uma configuração parametrizada, e a percepção é o dublê determinístico de `tests/end_to_end/chain.py` (`--dry-run-executors chain:dry_run_executors`, que marca o relatório como `dry_run`). O teste confere o relatório completo, a ausência de caminhos da máquina, a parada antes de qualquer `ContextRun` quando faltam os providers, a recusa de sobrescrever um run e a recusa de um run real de um checkout sujo. O dry run roda o script no lugar, numa árvore suja inclusive; os runs reais rodam o script de uma cópia deste bundle num repositório git temporário (`checkout_copy`, em `tests/end_to_end/incremental_harness.py`), que é o checkout que o script registra, então nenhum teste depende do estado da árvore de quem os roda. A regra da identidade do código é testada à parte, contra repositórios git temporários, em `tests/end_to_end/test_harness_code_identity.py`. Nada disso é evidência sobre dados reais.

## Lacunas encontradas ao preparar o harness (corrigidas na #659)

A preparação do harness encontrou três lacunas da CLI instalada, que o harness contornava compondo os estágios de jusante fora da CLI e injetando-os por `main(executors=...)`. A #659 as corrigiu na runtime, e o harness passou a usar só a CLI, sem mudar o protocolo: as mesmas seleções, os mesmos builds, a mesma configuração e a mesma identidade de código em cada run.

- **A CLI instalada não materializava um `ContextMap`.** Ela compunha fusão, Semantic Mapping, Entity Resolution e Spatial Relations sem `code_version`; os runs gravavam `code_version = ""`, e o estágio `context_map` recusava citá-los na linhagem (`code_version must not be empty`). Agora a CLI passa o `--code-identity` do comando, e sem ele esses estágios são bloqueados no preflight com a causa.
- **A CLI nunca compunha `semantic_mapping`**, que precisa de `semantic_map_id` e `code_digest`. Agora os dois são derivados pela runtime (`executor_identities`): o digest é o SHA-256 do `--code-identity`, a regra que o harness usava, e o mapa semântico é `semantic-map-<dataset>`, no lugar do `semantic-map-<experimento>-<run>` que o harness fornecia. Nenhuma métrica usa esse identificador: as entidades são pareadas pela chave geométrica.
- **O preflight da CLI exigia os módulos de DINOv2 e CLIP** mesmo quando um provider era declarado para eles, porque a CLI não repassava `provided_runtimes`. Agora repassa; o template não declara provider para DINOv2 nem para CLIP, então o extra `vision` continua necessário.

## Anexo: protocolo pré-registrado (cópia literal)

Comentário de 2026-09-26 na issue #501:

## Protocolo do braço real (pré-registro, antes da corrida)

**Decisão.** O backend semântico (Qwen3-VL-4B, nf4) é estocástico mesmo com greedy e `temperature=0.0`: entre duas execuções idênticas no v0.1.0, 29 de 90 claims coincidiram (32,2%). Essa variação é ruído de execução, não incerteza do mundo, e não pode ser confundida com efeito de partição. Por isso a comparação se divide em duas camadas, e o ruído de rerun é medido neste mesmo experimento. A dependência da #580 deixa de existir.

### Camadas

Levantadas no código de `milestone/v0-1-1` @ `ef140c8`, perfil `canonical/1`.

**Camada 1, exata.** Depende só de geometria e contagem; nenhum caminho do Qwen chega nela.
- Regiões aceitas por frame (bbox, kind, accepted). O SAM2 roda antes da interpretação e não depende dela (`visual_perception/pipeline.py`). Entram só como diagnóstico, para localizar uma divergência.
- Observações físicas, resultados de inferência e observações excluídas.
- Suportes da fusão, chaveados pelo conjunto de geometria. `geometry-jaccard-support-v1` usa só a identidade da geometria, nunca label, claim ou score (`semantic_fusion/support.py`).
- Contribuições: uma por região × frame × inferência.
- Entidades chaveadas pela geometria. `one-support-one-entity-v1` materializa toda saída da fusão, sem filtro semântico. Rejeições contadas por motivo.
- Entidades resolvidas. No perfil canônico só o canal geométrico está ativo: `semantic_compatibility`, `appearance`, `temporal` e `representation` não estão selecionados.
- Relações por (predicado, estado, chaves geométricas das entidades, tipos de incerteza). `spatial_relations` não lê estado semântico.

As features DINOv2/CLIP não entram em nenhuma decisão da camada 1 no perfil canônico. Um perfil que ligue `semantic_compatibility` ou `appearance` move resolução e relações para a camada 2.

**Camada 2, estatística.** Depende das claims.
- Claims por (frame, região): hipótese, papel, categoria e confiança. É a mesma chave canônica de `scripts-run5/15_visual_perception_content_equivalence.py`; `region-NNNN` é por frame e não depende do run.
- Na fusão: `claims.*`, `hypotheses`, `evidence_stances`, `abstaining_claims`, `uncertainty`, `supports_with_uncertainty` e `scorer_signals`.
- Nas entidades: `AmbiguityState` e o conjunto de labels das hipóteses.

A correspondência entre as entidades de execuções diferentes vem da camada 1 (chave geométrica), então a camada 2 compara entidade a entidade.

### Desenho

- **Dataset:** corridor-02, perfil `canonical/1` do cenário 1.0.5, 100 frames consecutivos `[s, s+100)`. O manifesto congela `s`, o commit, os checkpoints e a configuração efetiva antes da corrida.
- **Fundação:** uma única `SpatialFoundation` para todas as execuções.
- **Execuções de percepção (3):**
  - `W1` e `W2`: uma `ContextRun` cada, sobre os 100 frames;
  - `C`: 4 `ContextRun`s de 25 frames.
- **Builds:**
  - `W1`, `W2` e `C`;
  - `R = [W1, W2]`, que é o cenário 2;
  - cenário 3: dois builds sobre as `ContextRun`s de `C`, com uma política de jusante alterada.
- **Variável alterada:** a partição, 1 contra 4 `ContextRun`s. Ficam fixos a fundação, os frames, o perfil, os checkpoints e o commit.
- **Baseline:** `W1 × W2` mede o ruído de rerun sem partição.
- **Custo estimado:**
  - GPU: ~37 s/frame na RTX 3060 (no v0.1.0 foram 747 s para 20 frames), ou seja ~3 h para 300 frames-execução;
  - memória: o pico de RSS de `sensor_association` foi de 25 GB no v0.1.0.

### Critérios

**H1 (camada 1).** O estado da camada 1 de `C` é igual ao de `W1` e ao de `W2`, módulo identidades de execução.
- **Pré-condição:** `W1 == W2` na camada 1. Se falhar, a camada 1 não é exata sob rerun (por exemplo, máscara do SAM2 não determinística: o v0.1.0 não a comparou pixel a pixel). O resultado entra como achado, e qualquer reclassificação vira decisão nova, nunca tolerância silenciosa.
- Com a pré-condição satisfeita, qualquer `C ≠ W` é divergência causada pela partição e é tratada como bug até prova em contrário.

**H2 (camada 2).** A partição não reduz a concordância semântica além do ruído de rerun.
- **Métricas,** reportadas separadamente:
  - `M_claim` = |A ∩ B| / |A ∪ B| sobre as chaves canônicas de claim, a mesma definição dos 29/90;
  - `M_state` = fração das entidades pareadas com o mesmo `AmbiguityState`;
  - `M_label` = fração das entidades pareadas com o mesmo conjunto de labels.
- **Estatística:** `D = ½[agr(W1,C) + agr(W2,C)] − agr(W1,W2)`.
  - IC 95% por bootstrap pareado por cluster: frames para `M_claim`, entidades para `M_state` e `M_label`.
  - 10.000 reamostragens, com a seed registrada no manifesto.
- **Decisão por métrica:**
  - limite superior do IC abaixo de 0: **rejeita** a invariância. É diferença científica real e precisa ser investigada antes de fechar a milestone;
  - limite inferior do IC ≥ −δ, com δ = 0,10 de concordância absoluta: **equivalente dentro de δ**;
  - caso contrário: **inconclusivo**, registrado como limitação conhecida nas notas da v0.1.1.

**Cenário 2 (exato).** `R` mantém as `physical_observations` de `W1` e dobra os `inference_results`.

**Cenário 3 (exato, estrutural).** Nenhum estágio de contexto é recomputado, e as decisões de reuso ficam registradas. As identidades de build diferem e as saídas podem diferir.

### Limitações declaradas

- Com 3 execuções, o bootstrap captura a variação entre frames e entidades, mas não a variância da concordância entre execuções: há um único par `W1 × W2`.
- δ = 0,10 é uma tolerância escolhida antes da corrida, não derivada. Sem reference set (#581), não há como amarrá-la a impacto em qualidade.
- Este experimento não mede qualidade semântica.
