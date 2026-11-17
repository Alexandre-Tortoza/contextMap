# Workflow incremental real sobre uma fundação espacial (issue #504)

Execução de aceitação da milestone v0.1.1: sobre dados reais do corridor-02, um mapa contextual é estendido com observações novas sem trocar a fundação espacial e sem tocar nos snapshots anteriores.

```text
SF1 (selecionada)
 ├─ CR1 sobre S1 ─ CB1 = [CR1]      → ContextMap M1
 └─ CR2 sobre S2 ─ CB2 = [CR1, CR2] → ContextMap M2        (M1, CB1 e CR1 nunca mudam)
```

> **Estado: preparado, ainda não executado em dado real.** O driver, as checagens de cada invariante e o relatório estão implementados, e um dry run sobre a cadeia de CI percorre as seis fases na suíte (`tests/end_to_end/test_incremental_workflow_harness.py`). Este ambiente não tem GPU, checkpoints nem o corridor-02: não há número real a citar até o comando abaixo rodar na máquina com GPU e o resultado ser registrado aqui.

Esta issue é distinta da #501: lá se mede invariância à partição; aqui se aceita o ciclo de vida. Os dois experimentos usam os mesmos passos (`experiments/incremental-partition-invariance-20260927/scripts/steps.py`), a mesma CLI e a mesma configuração.

## Fases

| Fase | O que roda | Comando |
|---|---|---|
| **A**, fundação | seleciona SF1 (`--foundation-run`) e cria a branch sobre ela; registra identidade e artifacts da fundação e o custo de construí-la, lido do journal do run da fundação | `contextmap context branch create` |
| **B**, primeiro incremento | percepção e associação sobre S1 → CR1 | `contextmap context run` |
| **C**, primeiro build | CB1 sobre a revisão 1 → M1, validado no nível `FULL`; CR1, CB1 e M1 são fotografados (SHA-256 de cada arquivo) e seus registros relidos | `contextmap context build` |
| **D**, segundo incremento | percepção e associação sobre S2, depois e disjunta de S1, contra SF1 → CR2, com a decisão de reuso de cada estágio | `contextmap context run` |
| **E**, segundo build | CB2 sobre a revisão 2 (CR1 e CR2) → M2, validado no nível `FULL` | `contextmap context build` |
| **F**, imutabilidade | reabre CR1, CB1 e M1, compara cada arquivo com a fotografia de C, relê os registros e valida M1 de novo | leitura, sem escrita |

B a E são, cada um, um comando da CLI num processo novo, medido por `measure_child_process`, com o `--code-identity` do run: a CLI compõe todos os estágios da configuração e grava essa identidade nos runs. Os builds rodam sem `resources.providers`, então não carregam modelo de percepção.

## Invariantes

Cada um é um booleano em `report.json` → `invariants`; `accepted` é verdadeiro só com todos:

| Invariante da issue | Chave | Como é conferido |
|---|---|---|
| SF1 é idêntica para CR1 e CR2 | `same_foundation` | a identidade da fundação de CR1, CR2, CB1 e CB2 é a de SF1 |
| o mesmo `MapId` é reusado; nenhuma troca escondida do mapa | `same_map_id` | M1 e M2 citam o `map_id` do mapa geométrico de SF1 |
| State Estimation e Geometric Mapping não rodam de novo | `foundation_not_recomputed` | nas ContextRuns os dois estágios são `provided` (de `execution.json`); nos builds nenhum deles é executado |
| CR1 fica imutável depois de CR2 | `cr1_immutable` | nenhum arquivo do run de CR1 mudou, surgiu ou sumiu, e `context_run.json` relido é o mesmo |
| CB1 fica congelado depois de CR2 | `cb1_immutable` | o mesmo para o run de CB1 e `context_build.json` |
| M1 continua válido e inalterado depois de M2 | `m1_unchanged_and_valid` | M1 revalidado no nível `FULL` (`verified`), mesmo `ContextMapId` e mesma identidade de conteúdo |
| CB2 consome explicitamente CR1 e CR2 | `cb2_consumes_cr1_and_cr2` | os `context_run_ids` de CB2 são exatamente CR1 e CR2 |
| as contagens da fusão se explicam pelas observações selecionadas | `fusion_counts_explained` | todo frame de contribuição de M2 está em S1 ∪ S2, as observações físicas não passam de \|S1\| + \|S2\| e há um resultado de inferência por observação física (seleções disjuntas, uma inferência cada) |
| toda entidade e relação de M2 rastreia até artifacts persistidos | `m2_lineage_closes` | os artifacts independentes de regiões, features visuais, interpretação semântica e associação na linhagem da fusão de M2 são exatamente os de CR1 e CR2; a validação `FULL` confere o fechamento da linhagem por hash |
| a validação `FULL` do `ContextMapArtifact` final passa | `m2_validates` | M2 `verified`, sem problema de inventário nos runs lidos |

## Métricas por fase

`report.json` → `phases.<A..F>`, cada uma em separado:

- **tempo de parede** e **pico de RSS** do processo do comando (`step.wall_time_seconds`, `step.peak_rss_mb`); F é leitura e registra só o tempo;
- **artifacts produzidos ou reusados**: a disposição de cada estágio (`provided`, `executed`, `reused`, `recomputed`) e, nas ContextRuns, os artifacts com `produced`/`reused`;
- **armazenamento acrescentado**: bytes do run (`step.storage_bytes`) e, nos builds, bytes por artifact (`build.artifact_bytes`; o do `context_map` é o tamanho do artifact final);
- **cargas de modelo**: `model_load_sets` conta execuções do estágio visual como proxy de um conjunto de modelos composto por ContextRun; os estágios regional e semântico são registrados separadamente no runtime e nenhuma carga ocorre nos builds;
- nos builds: observações físicas, resultados de inferência, suportes, contribuições, observações excluídas, entidades, rejeitadas, entidades resolvidas, relações por estado e o status da validação;
- **custo evitado** (`cost_avoided`): o `elapsed_s` de `state_estimation` e `geometric_mapping` no journal do run da fundação, que um incremento não paga. Um estágio que o run da fundação recebeu pronto não tem custo registrado ali.

A issue não afirma que M2 é "melhor" que M1 por ter mais observações. Sem anotações para o corridor-02, a qualidade fica **N/A**: o relatório descreve só mudanças estruturais e de comportamento.

## Manifesto

`workflow-manifest.json` é gravado **antes** da fase A: commit e estado da árvore, identidade do código (num run real, o SHA completo do commit), dataset, o `SequenceArtifact` exato, a fundação (identidade e as três refs), o run da fundação (relativo ao workspace), S1 e S2 exatos, a configuração efetiva (documento e digest, sem o caminho do workspace), a configuração de cada backend de percepção (os checkpoints), os alvos de provider e o ambiente (`capture_environment`: versões, plataforma, dispositivo, pacotes opcionais; e as GPUs por `nvidia-smi`, quando houver). As identidades de todos os artifacts produzidos e reusados vão em `report.json`.

## Pré-requisitos e custo

Os mesmos do braço real da #501 ([README](../incremental-partition-invariance-20260927/README.md#pré-requisitos-na-máquina-com-gpu)): extra `vision`, providers de SAM2 e Qwen em `resources.providers` (cada um devolve só o runtime do modelo carregado; os adapters do run entregam a ele os pixels de cada pass e os bytes verificados de cada view, #661), a configuração a partir de [`config/corridor-02.template.json`](../incremental-partition-invariance-20260927/config/corridor-02.template.json) e um run concluído da fundação no workspace. Sem providers, a fase B para no preflight, antes de qualquer estágio, e o relatório sai `failed` com a causa; nada é fingido. Também como lá, o checkout é conferido pela mesma regra (`resolve_code_identity` do `steps.py` da #501): um run real é recusado com status 2, antes de criar qualquer diretório, sem um commit que identifique o checkout, com a árvore não comprovadamente limpa ou com um `--code-identity` que não nomeia exatamente o commit do checkout; a mensagem lista cada motivo e diz como corrigi-lo. O dry run é isento.

Custo esperado, pelos números do v0.1.0 que a #501 registra: ~37 s por imagem na RTX 3060, então ~1 h para S1 + S2 = 100 imagens (o padrão), e o pico de RSS de `sensor_association` foi de 25 GB. A issue não fixa S1 e S2; o padrão são duas janelas consecutivas de 50 imagens a partir de `--first-image`. Usar o mesmo `s` do braço da #501 põe M2 sobre as mesmas 100 imagens de lá.

## Comando

Da raiz do checkout, numa árvore limpa no commit que será citado:

```bash
python experiments/incremental-context-workflow-20260927/scripts/run_workflow.py \
    --workspace outputs/<workspace> \
    --foundation-run outputs/<workspace>/<dataset>/run-NNNN \
    -c corridor-02.json \
    --first-image <s>
```

`--s1-images` e `--s2-images` (padrão 50 cada) mudam as janelas; `--code-identity` omitido é o `HEAD` do checkout e, num run real, dado, tem de nomear exatamente o commit do checkout (um SHA abreviado dele é aceito): o que o manifesto e cada fase gravam é sempre o SHA completo desse commit, e no dry run o valor dado é gravado como veio; `--run-number` numera uma rodada nova. A execução é fail-fast: a primeira fase que falha encerra o run e `report.json` sai `failed`, com a fase e a causa. Só `"status": "complete"`, `"accepted": true` e status de saída 0 são evidência de aceitação. Um diretório de saída existente é recusado (status 2).

Saída em `outputs/incremental-context-workflow-20260927/run-NNNN/harness/`: `workflow-manifest.json`, `report.json` e `steps/` (pedido e resposta de cada comando, com caminhos da máquina, fora do Git). Os artifacts ficam no workspace, onde a runtime os publica.

## Ao registrar o run real

1. Copiar para este diretório `workflow-manifest.json` e `report.json`, e escrever o `manifest.json` do bundle com o SHA-256 completo de cada arquivo executado (este `scripts/` e os `scripts/` e `config/` da #501 que ele usa), como pede [`docs/ARTIFACTS.md`](../../docs/ARTIFACTS.md).
2. Resumir aqui cada fase (tempo, memória, armazenamento, reuso), os invariantes, o custo evitado e as falhas de backend registradas, sem inferir qualidade de contagens.

## Dry run na CI

`tests/end_to_end/test_incremental_workflow_harness.py` roda este comando sobre a fundação sintética, com a percepção substituída pelo dublê determinístico (`--dry-run-executors chain:dry_run_executors`, relatório marcado `dry_run`) e todo o resto composto da configuração. Confere as seis fases, todos os invariantes, a ausência de caminhos da máquina, a parada na fase B sem providers, a recusa de um run real de um checkout sujo e a detecção de um arquivo alterado pela fotografia de F. O dry run roda o script no lugar, numa árvore suja inclusive; os runs reais rodam o script de uma cópia deste bundle e dos scripts da #501 num repositório git temporário (`checkout_copy`, em `tests/end_to_end/incremental_harness.py`), que é o checkout que o script registra, então nenhum teste depende do estado da árvore de quem os roda.

As lacunas da CLI instalada que o harness contornava (nenhum `ContextMap` materializável sem `code_version`, `semantic_mapping` nunca composto, preflight sem `provided_runtimes`) foram corrigidas na #659, e cada fase agora é só um comando da CLI; estão descritas no [README da #501](../incremental-partition-invariance-20260927/README.md#lacunas-encontradas-ao-preparar-o-harness-corrigidas-na-659).
