# Reuso de estágios

Um artifact de estágio é imutável, então seu resultado pode ser reutilizado sempre que tudo de que ele depende é idêntico. O runtime decide isso estágio a estágio, em ordem de dependência, para que alterar um estágio invalide exatamente ele e quem depende de fato do que ele produz.

O reuso **nunca** usa nome de diretório nem nome legível de run como chave, e nunca esconde o artifact anterior atrás de um "cache hit" genérico: a decisão nomeia o artifact exato e o motivo.

## Chave de reuso

`ReuseKey` reúne tudo de que o resultado de um estágio depende:

| Componente | Origem |
|---|---|
| estágio e contrato de saída | `PlannedStage` |
| configuração própria do estágio | `PlannedStage.config_digest` (contrato declarado, backend e parâmetros dos componentes) |
| entradas | o tipo e o **hash de conteúdo** de cada artifact de entrada (`ArtifactRef.content_hash`), não seu nome nem seu id de run |
| código e políticas | `ReusePolicy.code_identity`, **sem default**: reusar entre versões de código é uma decisão |
| seleção de observações | para um estágio `observation_scoped` com `inputs.observation_selection`, a seleção entra no `config_digest` do estágio (#497); a jusante, ela chega pelo hash de conteúdo das entradas |
| identidades extras | `ReusePolicy.identities[estágio]`: calibração, mapa, transformações opcionais |
| fonte (estágio sem entradas) | `identities[estágio]["source"]`: o que o estágio lê de fora do DAG (ver abaixo) |
| configuração efetiva | `identities[estágio]["configuration"]`: defaults e algoritmos que os backends do executor resolvem sozinhos (ver abaixo) |

`ReuseKey.digest` é o SHA-256 da forma canônica. Como a chave usa o **conteúdo** das entradas, dois artifacts com o mesmo hash de conteúdo são intercambiáveis, e um estágio recomputado que reproduz o mesmo conteúdo não invalida seus dependentes.

O `content_hash` é declarado pelo executor que produz o artifact (por exemplo, o digest do inventário do manifest). Um artifact sem ele pode ser consumido, mas nunca reutilizado nem indexado: sem identidade de conteúdo, a chave dos dependentes não pode ser construída e eles são recomputados, com o motivo registrado. O `ArtifactRef` também carrega o `location` do artifact (relativo ao workspace): quando um estágio é reutilizado, o run novo guarda a **referência** ao diretório do run que o gravou, nunca uma cópia, e um estágio a jusante o abre por `StageRequest.directory_of`.

### Estágio-fonte

Um estágio sem entradas (`ingestion`, `pose_ingestion`) lê de fora do DAG: sua chave não teria nada além da configuração, e dois bags diferentes ingeridos com a mesma configuração compartilhariam a chave, reutilizando em silêncio o artifact do outro. Por isso a chave de um estágio-fonte **exige** a identidade da fonte, em `identities[estágio]["source"]`:

- um executor que lê a fonte a nomeia em `source_identity` (protocolo `SourceStageExecutor`); `IngestionStageExecutor` devolve `IngestionRequest.identity`. `run_plan()`, `preflight()` e `predict_reuse(..., executors=...)` incorporam essa identidade à política; a CLI faz o mesmo antes do dry-run (`with_executor_identities`);
- sem executor (um run totalmente reutilizado), a política declara a fonte explicitamente;
- sob reuso, um estágio-fonte sem identidade de fonte é **problema de preflight** (`reuse.identities.<estágio>.source`), e uma identidade declarada que contradiz a do executor também. Se o preflight for contornado, a decisão é `recomputed`, sem chave, com o motivo registrado.

O `artifact_id` publicado pelo `IngestionStageExecutor` combina a identidade do estágio com a da fonte, então fontes distintas nunca publicam o mesmo id.

### Configuração efetiva de um executor

`PlannedStage.config_digest` cobre os parâmetros que a configuração **declara**. Um backend também aplica o que a capability dona resolve sozinha: o default de cada parâmetro não declarado e a versão de cada algoritmo. Duas execuções que diferem só aí produzem artifacts diferentes e não podem compartilhar a chave (#636). O runtime não conhece esses defaults (a configuração, o DAG e o reuso nunca dependem de uma capability); quem os conhece é o executor, que compôs os backends:

- um executor cujos backends resolvem configuração própria a nomeia em `configuration_identity` (protocolo `ConfiguredStageExecutor`), e `run_plan()`, `preflight()`, `predict_reuse(..., executors=...)` e a CLI (`with_executor_identities`) a incorporam como `identities[estágio]["configuration"]`, exatamente como a fonte; o executor também a combina no `run_id` que publica;
- `RegionPerceptionExecutor` nomeia a provenance do backend de Region Discovery e a `RegionDiscoveryIdentity` dele (configurações de passes e normalização com os defaults resolvidos e versões dos algoritmos do discovery): mudar um default ou um algoritmo do discovery recomputa o produtor regional, e declarar um parâmetro com o valor do default não muda essa identidade;
- uma identidade de configuração declarada na política que contradiz a do executor é **problema de preflight** (`reuse.identities.<estágio>.configuration`).

Consequência: sem o executor composto, a chave de um estágio configurado não pode ser reconstruída, então ele não é reutilizado (a previsão o dá como recomputado e o preflight exige o executor), a menos que a política declare a identidade de configuração explicitamente. Reusar a percepção sem compor seus backends deixou de ser possível por omissão: era justamente o caso em que um default ou algoritmo diferente passaria despercebido.

## Índice

`FileArtifactStore` guarda uma entrada JSON pequena e imutável por identidade; não há serviço de cache global mutável.

- A entrada é escrita **somente depois** que o estágio conclui, de forma atômica e sem substituir uma entrada válida. Uma falha, uma interrupção ou uma saída parcial não deixam nada que satisfaça uma consulta; arquivos temporários (`.tmp-*`) nunca são entradas.
- `find()` revalida a entrada contra a chave, checa a forma e pergunta ao callback `verify` se o artifact ainda existe e está íntegro (por exemplo, abrindo-o com o leitor da capability). O `verify` **não tem default**: um índice não sabe se o artifact por trás de uma entrada ainda existe. Entrada ilegível, divergente, de outro estágio ou que falha na verificação é tratada como ausente.
- `record()` nunca substitui uma entrada válida (o primeiro artifact concluído de uma identidade permanece indexado) e repara uma entrada inválida com o novo artifact.

## Decisão por estágio

`ReuseDecision` acompanha cada estágio no `ExecutionRecord`:

- `reused`: o artifact exato reutilizado (`reused_from`) e o digest da chave;
- `recomputed`: o motivo. Sem artifact anterior com essa identidade; entrada ilegível ou divergente; artifact anterior que falhou na verificação; recomputação forçada; entrada sem hash de conteúdo (sem chave); e, quando aplicável, saída sem hash de conteúdo (não indexada) ou artifact anterior que permanece indexado.

Sem `ReusePolicy`, o runner sempre executa e não registra decisão.

## Invalidação sensível ao DAG

Os estágios são decididos em ordem topológica, com a chave calculada a partir das entradas reais (reutilizadas ou recém-produzidas):

- alterar uma política de Semantic Fusion recomputa só a fusão; ingestion, percepção, estado, geometria e associação são reutilizados;
- alterar descoberta/grounding/refinamento recomputa o produtor regional e consumidores de seu novo conteúdo; estado e geometria são reutilizados;
- alterar apenas a política semântica recomputa `semantic_interpretation` e a fusão, preservando regiões, features e associação;
- alterar apenas features recomputa `visual_perception` e consumidores do canal visual, preservando o produtor semântico que depende de `regions`;
- se uma recomputação reproduz o mesmo conteúdo, os dependentes seguem reutilizáveis;
- mudar `code_identity` invalida tudo, e uma identidade extra invalida o estágio para o qual foi declarada.

## Braços de um experimento

Um DAG alternativo continua identificável (seu plano tem outro digest) e compartilha o upstream idêntico. Com a inserção de um estágio opcional entre a extração densa e a associação, o braço com enhancement reutiliza os artifacts exatos de ingestão e extração do braço nativo, executa só o enhancement e a associação, e o braço nativo permanece reutilizável.

## Previsão e preflight

`predict_reuse()` diz, sem executar nada, o que seria reutilizado. A previsão é **conservadora**: um estágio a jusante de outro que será recomputado aparece como recomputado (suas entradas ainda não existem), embora a execução possa reutilizá-lo se a recomputação reproduzir o mesmo conteúdo; o registro da execução é a autoridade.

No `preflight()`, um estágio que **certamente** será reutilizado dispensa executor, módulos opcionais e segredos: nada dele vai rodar (a configuração continua precisando estar completa). Forçar um estágio fora da execução é um problema, assim como um estágio-fonte sem identidade de fonte (ver [Estágio-fonte](#estágio-fonte)).

## Recomputação forçada

`ReusePolicy.force_recompute` obriga um estágio a rodar mesmo com a identidade indexada. A decisão registra `forced recomputation requested`; o novo artifact vale para a própria execução e **não** desloca o indexado.

## Lacunas conhecidas

- Os executores reais precisam calcular o `content_hash` a partir dos manifests das capabilities e fornecer o `verify` com seus leitores; isso acompanha os executores da validação end-to-end (#177). Os testes usam estágios puros que derivam o conteúdo de suas entradas e configuração.
- O índice não tem coleta de lixo nem é distribuído: fora do escopo do v0.1.0.
- A retomada de um run interrompido usa exatamente estas checagens de reuso ([`lifecycle.md`](lifecycle.md)); a seleção de runs e a validação de linhagem estão em [`selection.md`](selection.md).
