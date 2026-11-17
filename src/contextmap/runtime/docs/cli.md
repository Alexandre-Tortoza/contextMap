# Linha de comando

A CLI é uma camada **fina** sobre o runtime: ela interpreta argumentos, chama os serviços e imprime o que voltou. Não há ramo por backend, lógica científica, parsing de dataset, matemática de projeção nem regra de fusão, entidade ou relação. Cada flag vira um **override de configuração**, então a CLI nunca é um segundo lugar que decide o que roda.

```text
contextmap ingest  --workspace DIR --source PATH --sequence-name NOME --topic CHAVE=TÓPICO ... [--timestamp-policy ARQ]
                   [--window-clock ID --window-start S --window-end S] [--calibration ARQ]
                   [--extra CHAVE=VALOR ...] [--preflight]
contextmap run     [opções] [--stage ESTÁGIO]... [--dry-run]
contextmap stage   ESTÁGIO [opções] [--dry-run]
contextmap inspect config | plan   [opções]
contextmap inspect artifact CAMINHO
contextmap validate CAMINHO [--full]
```

Também funciona como `python -m contextmap ...`. O ponto de entrada de console é `contextmap = contextmap.runtime.cli:main`.

## Opções de configuração

| Flag | Vira |
|---|---|
| `-c/--config ARQUIVO` (repetível) | camada de arquivo `.json`/`.toml`; a última vence |
| `--profile ID` | perfil base (padrão `canonical/1`) |
| `--set CAMINHO=VALOR` (repetível) | override direto |
| `--sequence ID` | `inputs.sequence` |
| `--select ESTÁGIO=REF[,REF...]` | `inputs.selections.<estágio>`: ids exatos, `latest` ou `named:<nome>` |
| `--catalog ARQUIVO` | catálogo dos runs disponíveis para selecionar |
| `--workspace DIR` | `resources.workspace` (o run store: datasets, runs e branches) |
| `--maps-root DIR` | `resources.maps_root` (diretório limpo onde um build publica cada mapa; ausente, o default é `<workspace>/../maps`) |
| `--device DEVICE` | `resources.device` |
| `--debug-level {none,standard,full}` | `policies.debug_level` |
| `--json` | saída legível por máquina |

Precedência: perfil < arquivos < `--set` < flags. A seleção continua explícita: um estágio sem `--select` nunca é escolhido implicitamente, e `--select` exige `--catalog` (o arquivo é uma entrada explícita; nada é descoberto por varredura de diretório). O formato do catálogo está em [`selection.md`](selection.md).

## Comandos

### `ingest`

Ingestion canônica pelo [serviço público de ingestion](ingestion-service.md): `--preflight` só confere o pedido; sem ele lê, valida, sincroniza e publica em `<workspace>/artifacts/ingestion-<dataset-name>/`; `dataset-name` é o valor de `--sequence-name`. Com fonte completa e hash habilitado (o padrão), uma segunda ingestão do mesmo ROS com a mesma configuração cria e valida primeiro um candidato; só então substitui esse diretório local. Se a nova ingestão falhar ou for interrompida, o artifact anterior permanece intacto. Se a fonte ou configuração for diferente, o comando recusa a substituição: use outro `--sequence-name`. `--workspace` é obrigatório. Num run do DAG, o estágio `ingestion` continua publicando em `<workspace>/artifacts/ingestion/<run>` pelo mesmo serviço. O adapter **não** é uma flag: vem do backend selecionado em `components.ingestion.source_adapter.backend` e é composto pela composition root (módulo opcional ausente falha com a dica de instalação). O progresso sai em stderr e o resultado em stdout (ou JSON, com os eventos); Ctrl+C sai com `130` sem publicar nada.

Campos do pedido que antes só a API Python preenchia, cada um uma flag explícita (sem ela, o pedido é o de antes, nada é inferido):

- `--timestamp-policy ARQ`: a política de timestamp (relógios e correção, por exemplo o `constant_offset` do corridor-02) no formato JSON que a proveniência da sequência grava (`encode_timestamp_policy`);
- `--window-clock ID --window-start S --window-end S`: a janela temporal da fonte, as três juntas ou nenhuma (uma janela parcial é erro de uso);
- `--calibration ARQ`: um `CalibrationSet` em JSON (`encode_calibration_set`, refinamentos incluídos), decodificado e validado antes de qualquer leitura da fonte e mesclado com a calibração que a fonte traz; um arquivo ilegível ou inválido é erro de uso;
- `--extra CHAVE=VALOR` (repetível): configuração específica do adapter, por exemplo `format=tum` e `pose_role=ground_truth` do `pose_file`.

### `run` e `stage`

Resolvem a configuração, derivam o plano, escopam (o pipeline completo, ou os alvos de `--stage`; `stage` é exatamente um alvo), validam o preflight e executam.

- **`--dry-run`**: mostra a configuração efetiva com o digest, a topologia resolvida (entradas, saídas e backends de cada estágio, o que roda, o que é fornecido por seleção, o que é indisponível e por quê), a seleção resolvida, o preflight e a cobertura de executores. **Não carrega modelo, não executa e não escreve nada.** Sai com `0` se o preflight passa e `1` se está bloqueado.
- **execução real**: exige `--workspace` (ou `resources.workspace`), porque persiste seus registros. Exige `inputs.sequence` (o dataset) e cria `<workspace>/<dataset>/run-NNNN/` **antes** de executar e o journal grava ali o plano, os eventos, o status e, ao concluir, `execution.json` ([`lifecycle.md`](lifecycle.md)). Cada run tem seu próprio diretório e nunca sobrescreve outro. Um preflight bloqueado, um estágio que falha ou uma interrupção deixam um registro inspecionável (`run record: <diretório>` na mensagem de erro); Ctrl+C sai com `130`.
- **identidade do código** (`--code-identity ID`, um commit ou uma versão): o journal do run a grava, o reuso a usa na chave e os estágios cujo run registra o código que o produziu (fusão, Semantic Mapping, Entity Resolution e Spatial Relations) a exigem. Sem ela, um plano que chega a esses estágios é bloqueado no preflight, antes de qualquer estágio, com `stages.<estágio>: no executor could be composed: stage '<estágio>' records code_version ...`; nenhum run grava um `code_version` vazio, e um `--code-identity` vazio é erro de uso (#659). Ver [Executores](#executores).
- **reuso e retomada** (`--reuse-index DIR`, `--code-identity ID`, `--force ESTÁGIO`, `--resume RUN`): o reuso combina a identidade exata dos estágios; `--resume` retoma um run falhado, cancelado ou interrompido como um run novo. Exigem um verificador de artifacts que o dono dos executores fornece a `main(verifier=...)`: a CLI não sabe se um artifact indexado ainda existe.

### `derive-spatial-support`

Deriva um `SemanticSpatialSupportArtifact`: junta as entidades de um `ContextMapArtifact` finalizado aos voxels ocupados de um `VoxelAggregationArtifact` do mesmo mapa bruto, de modo que a cor semântica sobreviva à simplificação da geometria para centroides. Os três artifacts a montante são selecionados explicitamente (`--context-map`, `--voxel-aggregation`, `--raw-geometry`) e apenas lidos; `--output` é criado atomicamente e nunca substituído, e `--run-id` é a identidade da derivação. Três artifacts que não compartilham um mapa bruto são recusados antes de qualquer leitura de geometria, com os três ids na mensagem. Mesmo com ids iguais, o comando verifica os hashes do `ContextMap` e os digests que prendem o `ContextMap` e a agregação ao mapa bruto exato. Upstream ausente, adulterado ou incompatível termina com código 1 e mensagem controlada, sem traceback.

Este comando existe porque `spatial_support` **não** é estágio do DAG: nada a jusante consome o artifact e o consumidor é o viewer, fora deste repositório. Ele não escreve registro de run nem consulta configuração — não aceita `--workspace` nem `--profile`. A agregação voxel que ele consome também não tem estágio: produzi-la continua fora do runtime.

### `inspect`

- `inspect config`: a configuração efetiva, o digest e as camadas que a produziram;
- `inspect plan`: a topologia resolvida, sem executar nada;
- `inspect run DIR [--events]`: o ciclo de vida de um run (estado, falha, problemas de bloqueio, estágios concluídos, eventos, ambiente);
- `inspect artifact CAMINHO`: um resumo do `manifest.json` de um diretório de artifact (id, schema, criação, número e tamanho dos arquivos; num `ContextMapArtifact`, também `context_map_id` e `content_identity`) e o resultado da checagem de integridade (a mesma do `validate`, no nível estrutural, com os avisos de um `ContextMapArtifact`), ou um documento do runtime.

### `validate CAMINHO [--full]`

Verifica a integridade: em um registro de run (`status.json`), o log de eventos (linha corrompida, lacuna na numeração), a coerência com o status e o digest de cada documento; em um diretório de artifact, cada arquivo do `file_inventory` (existência, tamanho e SHA-256; um caminho que sai do artifact é recusado); em `effective_config.json`, `plan.json` e `execution.json`, a versão de schema e o digest. Sai com `1` se algo falha e nomeia o arquivo ou o problema.

Um `ContextMapArtifact` (`artifact_type` `context_map` no manifest) passa, além disso, pelo validador da própria capability `artifact` ([`validate_context_map_artifact`](../../artifact/docs/integrity-validation.md)), que é quem sabe o que torna o mapa válido: um manifest editado à mão com o inventário coerente, por exemplo, é recusado com `manifest.identity_mismatch`. O nível padrão é o `STRUCTURAL` (manifest e versões, um `stat` por arquivo, descritores, estrutura dos índices, documentos pequenos e onde estão as dependências); `--full` roda o `FULL` (também cada registro, as referências, os índices reconstruídos e os arquivos das dependências a montante). Cada achado de severidade `error` entra em `integrity.problems` como `<código>: <mensagem>` e faz o comando sair com `1`. Um aviso (`warning`, como uma dependência opcional ausente ou sem hint gravado, que todo mapa escrito pelo writer traz para a sequência, o run de percepção e o semantic map) não invalida o mapa pelo contrato do validador: entra em `integrity.warnings`, no mesmo formato e na ordem do relatório, e na saída texto numa linha `warning: <código>: <mensagem>` depois dos problemas, mas **nunca** muda o código de saída, que continua definido só pelos erros (#635). `integrity.warnings` só existe para um `ContextMapArtifact`: os demais artifacts, os registros de run e os documentos do runtime não distinguem avisos e não mudam. As dependências são procuradas pelos hints relativos do manifest. `--full` não muda nada para os demais caminhos, que têm um único nível de verificação.

Só esse ramo carrega uma capability (`contextmap.artifact`, importada de forma tardia); os demais tipos de artifact, os registros de run e os documentos do runtime usam só a biblioteca padrão: não importam NumPy, modelo nem ROS.

### `context`

O contexto incremental (#502; decisão em [`runtime-composition.md`](../../../../docs/runtime-composition.md#contexto-incremental-v011-decisão-de-arquitetura)). Cada comando só compõe as APIs do runtime (`foundation_of_run`, `context_scope`, `publish_context_run`, `append_to_branch`, `plan_context_build`, `publish_context_build`); nenhum decide ciência.

```bash
# um run que produziu (ou recebeu) sequência, trajetória e mapa vira a fundação de uma branch
contextmap context branch create corredor --from-run run-0001 -c exp.toml --workspace ws
# cada ContextRun roda percepção e associação sobre inputs.observation_selection
contextmap context run --branch corredor --code-identity v0.1.1 -c exp.toml --workspace ws \
  --set 'inputs.observation_selection={"kind": "frame_range", "start_frame_index": 0, "end_frame_index": 500}'
# um build congela uma revisão e materializa o ContextMap
contextmap context build --branch corredor --code-identity v0.1.1 -c exp.toml --workspace ws
contextmap context build --branch corredor --revision 1 --code-identity v0.1.1 -c exp.toml --workspace ws
# inspeção só lê registros: não carrega modelo nem capability
contextmap context inspect branch corredor -c exp.toml --workspace ws
contextmap context inspect run ws/S1/run-0003 --workspace ws
```

- `branch create NOME --from-run RUN` valida como uma fundação a sequência, a trajetória e o mapa que o registro de execução do run nomeia, produzidos ou recebidos, e grava `branch.json`.
- `run --branch NOME` escopa o plano aos estágios de contexto com a fundação fornecida (nunca recomputada), executa, publica `context_run.json` e acrescenta a `ContextRun` como a próxima revisão. A saída diz, por estágio de contexto, `produced`, `reused` ou `not selected`.
- `run --branch NOME --select region_perception=<id> --select visual_perception=<id> --select semantic_interpretation=<id> --catalog FILE` reutiliza os três artifacts explicitamente. Só os stages não selecionados, como associação e representação opcional, executam. A `ContextRun` registra cada produtor como `reused`. Antes de executar, manifests devem cobrir a sequência da fundação e exatamente `inputs.observation_selection`; visual/semântico devem citar o produtor regional selecionado. Cada consumidor pode ser selecionado separadamente, sempre junto com sua seleção regional explícita. Stages de fundação vêm da branch; selecionar associação ou outro stage fora desses três é erro de uso.

- `build --branch NOME [--revision N] [--member ID ...] --code-identity ID` congela a revisão (ou um subconjunto explícito dela), grava `context_build.json` antes do primeiro estágio e roda da fusão até `context_map`, com todos esses estágios compostos pela própria CLI: o binário instalado materializa e verifica o `ContextMapArtifact` sem executor injetado (#659). Imprime o `ContextBuildId` e o `ContextMapId`. Sem `--code-identity` é erro de uso: a identidade do build inclui o código que materializa o mapa, e os runs da fusão em diante o gravam. O `context_map` é sempre publicado isolado por `map_id` em `<maps root>/<map_id>/<NNNNNN>` (o `map_id` é o nome da branch), um diretório imutável por build, e o build é acrescentado ao índice `branches/<map_id>/builds/`; a saída ganha `map` (índice e local). O maps root vem de `--maps-root DIR` ou `resources.maps_root`; ausente, a CLI usa o default `<workspace>/../maps`, um irmão limpo do run store. Publicar recusa, com erro de uso, um `<map_id>/<NNNNNN>` que já exista de um build interrompido.
- `inspect branch NOME [--revision N]` e `inspect run RUN --workspace DIR` só leem JSON; `inspect branch` também lista os builds do mapa (índice, `ContextMapId` e local).

Os erros de fundação, branch, `ContextRun` e build saem com código `1` e a mensagem do runtime; `--reuse-index`, `--code-identity` e `--force` valem para `run` e `build` como no `run` comum (em `context run`, `--code-identity` é opcional e, dado, vai para o journal e para o run da associação). Os executores também são compostos como no `run` comum (ver [Executores](#executores)): um estágio que a composição não conseguiu montar, como a percepção sem provider, bloqueia `context run` ou `context build` com o motivo, no componente em que falhou (#602).

## Erros acionáveis

Todo problema sai com o caminho e o que fazer: perfil desconhecido (lista os conhecidos), configuração inválida (todos os problemas de uma vez), módulo opcional ausente (com a dica de instalação), segredo ausente (pelo nome), seleção incompatível (o que diverge, com os ids), estágio sem capability, estágio sem executor. Com `--json`, uma falha sai como `{"ok": false, "error": ..., "problems": [...]}` na saída padrão.

Códigos de saída: `0` sucesso; `1` pedido entendido, mas não atendível (configuração inválida, preflight bloqueado, estágio que falha, integridade); `2` uso incorreto da linha de comando (por exemplo, execução real sem workspace).

## Fronteira

A CLI não faz rede, não abre dashboard e não usa "todos os runs". Importar `contextmap.runtime.cli` não carrega SDK pesado (torch, transformers, rosbags, NumPy); um teste garante isso. A CLI importa exatamente duas capabilities, cada uma dentro de uma função: a raiz pública de `contextmap.artifact`, no ramo do `ContextMapArtifact` de `validate` e `inspect artifact` (`_context_map_findings`), e a raiz pública de `contextmap.spatial_support`, no handler de `derive-spatial-support` (`_derive_spatial_support`). As duas exceções são da função, não do módulo: o mesmo import em qualquer outro ponto da CLI é violação (`tests/architecture/test_runtime_boundaries.py`).

## Executores

Para `run`, `stage`, `context run`, `context build` e seus dry runs, a CLI monta automaticamente os executores que [`compose_executors`](composition.md) consegue construir a partir da configuração resolvida e do próprio comando — hoje `state_estimation`, `geometric_mapping`, `sensor_association` e `context_map`; com `--code-identity`, também `semantic_fusion` (acumulação baseline), `semantic_mapping`, `entity_resolution` e `spatial_relations`, e `point_representation` quando a configuração habilita o estágio opcional (#663; o `ptv3` precisa ainda de um provider de `PTv3Runtime`); e, quando cada backend selecionado que não tem loader empacotado (`sam2`/`sam3`/`florence2` em region discovery, `qwen`/`gemini`/`florence2` em semantic interpretation) tem um `RuntimeProvider` disponível, também `region_perception` e `semantic_interpretation`, cada um independentemente; `visual_perception` compõe seus próprios extratores. O que os runs a jusante gravam e nenhuma configuração carrega vem de [`executor_identities`](composition.md#identidades-que-um-run-grava-executor_identities), a mesma regra da fachada `Runtime`: `code_version` é o `--code-identity`, `code_digest` é o SHA-256 dele e `semantic_map_id` é `semantic-map-<dataset>` (#659). Sem `--code-identity`, esses estágios ficam sem executor e o preflight de um run real diz por quê, em `stages.<estágio>`; um dry run os lista em `executors.missing`. Esse provider chega por **dois** caminhos, e a CLI não inventa um terceiro: `main(argv, providers=...)` (só um chamador Python: um teste, uma API embutida) ou `resources.providers` **dentro da própria configuração resolvida** (`"<capability>.<slot>" -> "módulo:atributo"`, resolvido por `resolve_provider`; ver [`composition.md`](composition.md#providers-declarados-em-configuração-resourcesproviders) e [`configuration.md`](configuration.md#resourcesproviders-runtime-de-modelo-declarado-em-configuração)). O segundo é o que fecha a lacuna real: o binário `contextmap` instalado, chamado como `contextmap run ...`/`contextmap stage ...`, nunca invoca `main()` com um `providers=` Python — só o primeiro caminho nunca bastaria para ele. Um `providers=` explícito, quando presente para o mesmo componente, ainda vence sobre um alvo declarado. O resultado composto se mescla com o que `main(argv, executors=...)` recebeu de quem chama; o que vem de `main()` sempre vence (um teste, um estágio que a composição não sabe montar como `ingestion`, ou uma substituição explícita). Isso é o que deixa o binário `contextmap` instalado compor essas capabilities sem um wrapper Python que monte o executor **ou o provider** à mão.

Um componente cujo runtime um provider supre (por `main(providers=...)` ou declarado em `resources.providers`) não carrega o loader empacotado, então o preflight também não exige os módulos opcionais dele: a CLI passa esses componentes como `provided_runtimes` ao preflight de todo comando que roda ou prevê um plano, pela mesma regra da composição ([`provided_runtimes`](composition.md#providers-declarados-em-configuração-resourcesproviders)). Um provider declarado para `visual_features.dense` (DINOv2) e `visual_features.region` (CLIP), por exemplo, dispensa torch, transformers e Pillow no preflight (#659); os segredos continuam exigidos.

`ingestion` continua sem ser composto automaticamente: o `IngestionStageExecutor` precisa de um `IngestionRequest` concreto (fonte, tópicos, sincronização) que não é parte de nenhuma configuração — é exatamente o que os flags do comando `ingest` constroem. O caminho canônico sem wrapper é `contextmap ingest` seguido de `contextmap run --select ingestion=<artifact-id> --catalog ...`, que trata a saída da ingestion como entrada fornecida em vez de reexecutá-la. Sem uma injeção, o preflight de uma execução real que precisa de `ingestion` bloqueia com `no executor is registered for it`, e nada roda; um dry-run não precisa de executores e informa quais faltam. Um estágio que a composição tentou e não conseguiu montar bloqueia a execução real com o motivo, no componente em que falhou (`components.<capability>.<slot>`: parâmetro recusado pelo backend, runtime de modelo ausente, alvo `resources.providers` inválido), em vez do genérico (issue #602).

## Lacunas conhecidas

- **`ingestion` dentro de `run`/`stage`.** Não é composto automaticamente (ver acima); precisa de injeção explícita ou de rodar como um `contextmap ingest` separado e ser fornecido/selecionado.
- **Providers de percepção ausentes.** Backends de descoberta regional e interpretação semântica sem loader empacotado exigem `main(providers=...)` ou `resources.providers` no respectivo componente. A ausência bloqueia o stage dono no preflight. O produtor visual compõe suas features independentemente.

- **Composição de estágios fora do plano.** A CLI compõe todo estágio que a configuração permite, não só os do plano escopado: com `resources.providers` declarados, um `context build` (que não roda percepção) ainda chama os providers da percepção ao compor. Quem não quer carregar modelo num build roda-o com `--set 'resources.providers={}'`, como fazem os harnesses de #501 e #504.
- **`point_representation` sem consumidor.** O estágio roda quando habilitado (#663), e `context run` grava o run na `ContextRun`, mas nenhum estágio o consome (#657): `context build` não o fornece à materialização, e o `ContextMap` não o cita.
- **`export`.** O estágio `context_map` já monta o `ContextMapArtifact`, e `contextmap.artifact` expõe `export_bundle`/`verify_bundle` na API pública, mas nada no runtime os consome: o parser não declara `export`, e a CLI só abre o `ContextMapArtifact` para validá-lo (`validate` e `inspect artifact`, ver acima). Oferecer o comando é uma decisão de produto ainda não tomada, não uma falta de artifact.
- **Verificador de artifacts.** As flags de reuso e retomada existem, mas o `verify` do índice vem do dono dos executores reais; a CLI recusa `--reuse-index` sem ele.
- **API pública.** A CLI chama as funções do runtime diretamente; a fachada pública frontend-neutra (`Runtime`, em `api.py`) passa a ser o que ela consome.
