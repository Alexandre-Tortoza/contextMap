# Configuração do runtime

A configuração diz **o que executar e com quais implementações**; nunca diz como uma capability funciona. Ela separa as preocupações que a arquitetura mantém distintas e resolve as camadas de forma determinística.

## Documento

Arquivos `.json` ou `.toml` (não há YAML: seria uma dependência nova sem consumidor). Todas as seções são opcionais em um arquivo; o perfil fornece o restante.

```toml
[pipeline]
preset = "canonical/1"

[pipeline.stages]            # topologia: só estágios opcionais podem ser desligados/ligados
point_representation = true

[components.region_perception.discovery]   # seleção de backend + parâmetros do backend
backend = "sam3"
[components.region_perception.discovery.sam3]
checkpoint = "..."
model_version = "..."         # obrigatório: entra na provenance e no digest
strategy = "text_prompt"
prompts = ["chair", "table"]  # conjunto de conceitos ordenado de um run

[components.region_perception.discovery.florence2]   # bloco de outro backend: ignorado
checkpoint = "..."

[inputs]                     # seleção/run inputs
sequence = "seq-01"
[inputs.observation_selection]   # opcional: só estes frames nos estágios de contexto
kind = "frame_range"
start_frame_index = 0
end_frame_index = 500
[inputs.selections]
state_estimation = "seq-01--run-0003"

[resources]                  # recursos/dispositivo
device = "cuda"
workspace = "outputs/runstore"  # run store: datasets, runs (run-NNNN) e branches
maps_root = "outputs/maps"      # opcional: só mapas; um context build publica cada mapa aqui
[resources.providers]        # RuntimeProvider declarado por componente (ver seção própria)
"region_perception.discovery" = "meu_pkg.contextmap_loaders:load_sam3"

[policies]                   # política de debug
debug_level = "standard"     # none | standard | full
```

| Seção | Preocupação |
|---|---|
| `pipeline` | preset de topologia e quais estágios opcionais participam |
| `components` | backend escolhido para cada ponto de variação, com os parâmetros **somente desse backend** |
| `inputs` | sequência esperada, seleção explícita de runs/artifacts upstream por estágio (ids exatos, listas de runs, seleções nomeadas ou `latest`, [`selection.md`](selection.md)) e, opcionalmente, `observation_selection`: as observações da sequência que os estágios de contexto processam |
| `resources` | dispositivo (repassado aos backends que declaram um parâmetro de dispositivo e não o definiram), `workspace` (o run store: datasets, runs e branches), `maps_root` (opcional: o diretório limpo em que um `context build` publica o `ContextMapArtifact` de cada mapa, em `<maps_root>/<map_id>/<NNNNNN>`, para um viewer escanear só ele; deixado ausente, o build ainda publica por `map_id` no default `<workspace>/../maps`, um irmão limpo do run store resolvido pela CLI) e `providers` (alvos `RuntimeProvider` declarados por componente, ver seção própria) |
| `policies` | nível de debug; `compose_executors` traduz `debug_level` no tipo próprio de cada capacidade e o repassa a todo writer de estágio que suporta debug (a percepção recebe os níveis de feature e semântico); o debug nunca é dependência contratual de um estágio downstream |
| `image_preparation` | máscaras estáticas por câmera (`valid_region` e `exclusions` nomeadas) aplicadas na preparação de imagem da percepção e da associação (ver seção própria) |

Não há campos de política de avaliação: nenhum consumidor existe ainda, e o schema não ganha campo sem consumidor.

### `image_preparation`: máscaras estáticas por câmera

Restrições espaciais de preparação de imagem, por câmera (chave = `calibration_id`), no espaço
da imagem preparada. Servem para regiões que ocupam os mesmos pixels em todo frame — o rig de
gravação e a borda preta de um fisheye. Ausente para uma câmera, nada é restringido (o
comportamento anterior).

```toml
[image_preparation."camera_1_image_raw-calib"]
valid_region = "assets/corridor-02/camera_1-valid.npy"   # foreground = pixels utilizáveis
exclusions.ego = "assets/corridor-02/camera_1-ego.npy"   # foreground = pixels excluídos (nome livre)
```

- Os assets são arrays booleanos 2D em `.npy` (Pillow não é dependência; NumPy é), no tamanho da
  imagem preparada; uma máscara de outra dimensão é recusada, nunca reamostrada.
- Referências relativas resolvem contra o diretório de trabalho atual; caminhos absolutos valem
  como dados.
- São decodificadas uma vez (`runtime/image_masks.py`) e entregues aos produtores que preparam
  imagens: em `region_perception`, uma proposta fora do válido ou dentro de uma exclusão é
  rejeitada na normalização (o rig nunca é segmentado/rotulado); em `sensor_association`, um
  ponto que projeta ali vira `OUTSIDE_VALID_SUPPORT` e não sustenta região.
- Entram no `to_document` e, portanto, no digest e na chave de reuso: mudar uma máscara é uma
  decisão registrada que gera outra identidade de run.
- Ligar `valid_region`/`exclusions` normalmente exige afrouxar os limiares de normalização do
  backend de discovery (`minimum_valid_fraction`, `maximum_exclusion_fraction`), senão o default
  `minimum_valid_fraction=1.0` rejeita por inteiro qualquer região que apenas encoste na vinheta.
  Ver o overlay `examples/v0.2.0/corridor-02-image-masks.toml`.

### `inputs.observation_selection`

Uma `SequenceSelection` na forma canônica de `contextmap.ingestion.encode_selection` (`frame_range`, `timestamp_range`, `explicit_ids`). Omitida, vale a sequência inteira; é assim que a sequência inteira se escreve, e `{"kind": "full"}` explícito seria só outra identidade para o mesmo trabalho.

- Vale para os estágios `observation_scoped` do catálogo: `region_perception`,
  `visual_perception`, `semantic_interpretation` e, quando habilitado, `semantic_scoring`. Os estágios da fundação espacial (`state_estimation`, `geometric_mapping`) continuam na sequência inteira; `sensor_association` cobre exatamente os frames do artifact regional que consome e grava a seleção dele.
- Entra no `config_digest` do estágio e, portanto, no `run_id` publicado e na chave de reuso. Sem seleção, o digest é o mesmo de antes do campo existir.
- A configuração só confere o formato (objeto com `kind` em texto). O executor decodifica pela Ingestion, exige a forma canônica (recusa campos de outro tipo de seleção, que uma fusão de camadas pode deixar) e resolve contra a sequência antes de preparar qualquer imagem.

## Precedência

Do menor para o maior, o último vence:

1. os padrões do **perfil** (`canonical/1`);
2. os **arquivos**, na ordem informada;
3. os **overrides** `dotted.path=valor`, na ordem informada.

Mapeamentos se mesclam chave a chave; qualquer outro valor substitui. O valor de um override é lido como literal JSON quando é um (`true`, `12`, `[1,2]`, `null`) e como texto caso contrário. O ambiente **não participa** da mescla: ele só carrega segredos.

## Configuração efetiva

`resolve_effective_config()` devolve uma `EffectiveConfig`. A resolução:

- valida a estrutura e devolve **todos** os problemas de uma vez (`ConfigurationError.problems`);
- lista todo estágio do preset com seu estado resolvido;
- mantém, por ponto de variação, apenas o bloco de parâmetros do **backend selecionado**: o bloco de outro backend não vaza para a execução nem entra no digest, o que permite trocar de backend por override sem editar o arquivo;
- descarta os componentes de um estágio desligado: um estágio não selecionado não é construído, então sua configuração não conta;
- repassa `resources.device` aos backends que o declaram, sem sobrescrever um valor explícito;
- rejeita combinações incompatíveis (hoje: o canal de evidência `point_representation` da acumulação de Semantic Fusion, que nenhuma configuração alimenta: a fusão não tem entrada `representation`, um contrato de consumo ainda fora de escopo; ligar o estágio `point_representation` não o alimenta).

O runtime **não escolhe backend em nome do usuário**: o perfil `canonical/1` fixa a topologia e deixa todo backend não selecionado. `check_selection()` reporta o que falta escolher.

## `resources.providers`: runtime de modelo declarado em configuração

Um backend sem loader empacotado (SAM2, SAM3, Qwen, Gemini, Florence-2 hoje) precisa de um `RuntimeProvider` — um `Callable[[config, ResolvedSecrets], runtime]` — para ser composto. Um chamador Python pode montar esse `Callable` diretamente (`Runtime(providers=...)`, `main(providers=...)`), mas o binário `contextmap` **instalado** não tem como: ele só enxerga o que a configuração descreve. `resources.providers` é essa descrição: um mapeamento `component_id -> "módulo:atributo"`, um por ponto de variação que precisa de um provider.

```json
{"resources": {"providers": {"region_perception.discovery": "meu_pkg.loaders:load_sam3"}}}
```

- **Validação aqui é só estrutural.** A resolução do documento (`resolve_effective_config()`) exige apenas que cada valor seja uma string não vazia; ela nunca importa o módulo. Importar e resolver o alvo em um `RuntimeProvider` de verdade é responsabilidade de `contextmap.runtime.composition.resolve_provider`, chamado só quando aquele componente está sendo composto de fato — ver [`composition.md`](composition.md#providers-declarados-em-configuração-resourcesproviders) para a precedência (um `providers=` explícito, quando existe, ainda vence), a preguiça e a postura de segurança.
- **Camadas e digest.** `resources.providers` flui pelas mesmas três camadas de qualquer outro campo (perfil < arquivos, mesclados chave a chave < overrides, que substituem o mapa inteiro) e participa do `digest` automaticamente, porque é só mais um campo de `RuntimeConfig.to_document()` — nenhum tratamento especial foi necessário.
- **Não é parâmetro de backend.** Um alvo declarado é uma decisão de composição/implantação (qual processo fornece qual runtime), nunca um parâmetro científico validado pela capability; por isso vive em `resources`, ao lado de `device`/`workspace`, e nunca dentro do bloco `components.<capability>.<slot>.<backend>` do próprio backend.
- **Segurança.** Um alvo é código Python executado em tempo de execução (importado e depois chamado). Configuração de origem não confiável nunca deve declarar um alvo, exatamente como não deve apontar para qualquer outro código executável — essa fronteira de confiança já existe hoje e não é nova.

## Digest e persistência

`digest` é o SHA-256 da forma JSON canônica (chaves ordenadas, sem espaços) do documento efetivo. Duas configurações que resolvem para os mesmos valores têm o mesmo digest, independentemente de quais arquivos, formatos ou overrides as produziram. `sources` registra as camadas (perfil, arquivos com hash do conteúdo, e a chave de cada override, nunca o valor).

`write_effective_config()` grava `effective_config.json` com `schema_version`, `digest`, `config` e `sources`. A escrita é atômica e nunca substitui um arquivo existente: repetir a mesma configuração é um no-op; uma configuração diferente é recusada, porque um run publicado é imutável. `read_effective_config()` recusa outra versão de schema, digest divergente e documento não resolvido.

## Segredos

Um segredo nunca faz parte da configuração:

- nomes de parâmetro que denotam credenciais (`api_key`, `password`, `authToken`, ...) são rejeitados na validação, com a orientação de usar o ambiente; `max_new_tokens` ou `keyframe_stride` não são confundidos com credenciais;
- cada backend declara no catálogo os nomes das variáveis de ambiente de que precisa;
- `resolve_secrets()` lê apenas esses nomes; `ResolvedSecrets` só entrega um valor por `get()`, e `repr`/`str` mostram nomes, nunca valores.
- Sem passar `environ=` explicitamente (o caso do binário `contextmap` instalado), `main()` completa `os.environ` com os pares `KEY=VALUE` de um `.env` na raiz do processo, se existir, antes de resolver qualquer segredo — uma variável já presente no ambiente real nunca é sobrescrita pelo arquivo. Veja `.env.example`. Um `environ=` explícito (todo teste, e qualquer embarcador Python) pula essa etapa por completo.

## Três níveis de verificação

| Função | Pergunta | Precisa de |
|---|---|---|
| `resolve_effective_config()` | o documento é válido e coerente? | nada |
| `check_selection()` | todo ponto de variação de um estágio habilitado tem backend? | nada |
| `check_availability()` | os módulos opcionais e os segredos dos backends selecionados existem? | consulta de módulos (sem importá-los) e ambiente |

Nenhuma delas carrega modelo. É o que permite um dry-run e uma inspeção de topologia antes de qualquer execução pesada.

## Módulos

A configuração é o subpacote `contextmap.runtime.configuration`, dividido pelas razões de mudança; o resto do runtime importa só o `__init__` dele.

| Módulo | Responsabilidade |
|---|---|
| `problems.py` | `ConfigProblem` e `ConfigurationError`, o vocabulário de problemas que também a seleção, o preflight e o serviço de ingestion usam |
| `schema.py` | as dataclasses tipadas, `CONFIG_SCHEMA_VERSION`, os vocabulários de política e `to_document()`, a serialização que o digest cobre; os defaults de política são os defaults dos campos de `PoliciesConfig` e de nenhum outro lugar |
| `documents.py` | a camada não tipada: o documento do perfil, a leitura de `.json`/`.toml`, a mescla chave a chave e a linguagem de overrides |
| `resolution.py` | `resolve_effective_config()`: validação estrutural, normalização, aplicação dos defaults e o digest; também a gramática de seleção (`LATEST`, `NAMED_PREFIX`) |
| `validation.py` | `check_selection()`, `check_availability()` e as regras entre capabilities que o runtime ainda hospeda (alcance da associação, canais da fusão), cada uma com um `FIXME` que nomeia o dono real |
| `secrets.py` | a heurística de nome de credencial e `ResolvedSecrets`/`resolve_secrets()` |
| `persistence.py` | `write_effective_config()` e `read_effective_config()` |

## Catálogo

O catálogo de opções (`contextmap.runtime.components`) é dado puro: nomeia estágios, pontos de variação e backends, e declara o que cada backend precisa (módulos opcionais do código **empacotado**, segredos, parâmetro de dispositivo). Um backend cujo runtime o chamador fornece não lista módulos: os do runtime fornecido são do chamador. Ele não importa backend, não carrega modelo e não define valor científico. Os parâmetros de um backend, como checkpoint e limiares, são validados por quem os possui; a composition root os valida pela configuração da própria capability e os passa a ela ([`composition.md`](composition.md)).

Pontos de variação:

| Componente | Backends |
|---|---|
| `ingestion.source_adapter` | `ros1_bag`, `ros2_bag` |
| `region_perception.discovery` | `sam2`, `sam3`, `florence2` |
| `region_perception.refinement` | `sam2`, **opcional**; exige `region_perception.grounding` selecionado |
| `region_perception.grounding` | `locateanything`, **opcional**: sem backend selecionado, o estágio roda sem grounding; selecionado, exige o grupo reservado `query_set` |
| `visual_features.dense` | `dinov2`, `dinov3`, `siglip2` |
| `visual_features.region` | `clip`, `alphaclip` |
| `semantic_interpretation.interpreter` | `qwen`, `gemini`, `florence2`, `eagle2_5` |
| `state_estimation.estimator` | `external_pose`, `fast_lio` |
| `semantic_scoring.scorer` | `clip`, `alphaclip`; do estágio opcional `semantic_scoring`, obrigatório quando ele está habilitado; `text_template` (com `{hypothesis}` uma única vez) é obrigatório, e o checkpoint e a revisão (ou o fingerprint dos pesos AlphaCLIP e o `visual_code_version`) precisam ser os das features de região pontuadas |
| `sensor_association.dense_sampling` | `dense-feature-sampling-v1`, **opcional**: sem backend selecionado, a associação é a geometry-only; selecionado, exige `interpolation` (`nearest` ou `bilinear`) e amostra um canal por feature densa do artifact visual selecionado (#651) |
| `point_representation.center_selection` | `observation-support-anchor-v1`; do estágio opcional `point_representation`, obrigatório quando ele está habilitado ([seleção de centros](../../point_representation/docs/center-selection.md)) |
| `point_representation.encoder` | `geometric_descriptor`, `ptv3`; do estágio opcional `point_representation`, obrigatório quando ele está habilitado; o grupo reservado `support_policy` é obrigatório |
| `semantic_fusion.support` | `geometry-jaccard-support-v1` |
| `semantic_fusion.accumulation` | `baseline-evidence-accumulation-v1`, `quality-aware-evidence-accumulation-v1` |
| `entity_resolution.retrieval`, `.resolution`, `.geometry_comparison` | políticas versionadas, obrigatórias |
| `entity_resolution.semantic_compatibility`, `.temporal_compatibility`, `.appearance`, `.representation` | políticas versionadas, **opcionais**: sem backend selecionado, o canal é `None`, nunca um padrão |
| `spatial_relations.frame_conventions`, `.candidate`, `.geometry_summary` | políticas versionadas, obrigatórias |
| `spatial_relations.geometric_predicate`, `.contact_predicate` | políticas versionadas, **opcionais**: sem backend selecionado, o avaliador é `None` |

Para uma política, o identificador de backend é a identidade que a própria capability já versiona.

Estágios de `canonical/1`, do recorded source ao `ContextMapArtifact`: `ingestion`,
`pose_ingestion` (opcional), `region_perception`, `visual_perception`,
`semantic_interpretation`, `semantic_scoring` (opcional), `state_estimation`,
`geometric_mapping`, `sensor_association`, `point_representation` (opcional),
`semantic_fusion`, `semantic_mapping`, `entity_resolution`, `spatial_relations` e `context_map`.
Os três produtores de evidência têm configurações e artifacts independentes:

| Stage | Configuração própria | Artifact |
|---|---|---|
| `region_perception` | descoberta, grounding e refinamento opcionais | `RegionPerceptionRunArtifact` |
| `visual_perception` | `visual_features.dense` e `visual_features.region` | `VisualEvidenceArtifact` |
| `semantic_interpretation` | intérprete e política de request/views | `SemanticInterpretationArtifact` |

Features e interpretação selecionam o produtor regional por `regions`; nenhum deles
reexecuta sua descoberta. `semantic_scoring` recebe `regions`, `visual_evidence` e
`semantic_interpretation`, e publica scores que a fusão pode consumir explicitamente.
`point_representation` recebe geometria e associação e publica uma saída derivada sem
consumidor canônico. Detalhes de entradas em [pipeline](pipeline.md).

## Ligar `point_representation`

O estágio fica desligado em `canonical/1`, e ligá-lo não muda nenhum outro estágio: nenhum consome o run. Para rodá-lo, habilite o estágio e selecione os dois pontos de variação dele:

```json
{
  "pipeline": {"stages": {"point_representation": true}},
  "components": {
    "point_representation": {
      "center_selection": {"backend": "observation-support-anchor-v1"},
      "encoder": {
        "backend": "geometric_descriptor",
        "geometric_descriptor": {
          "support_policy": {
            "support_type": "neighborhood",
            "method": "radius",
            "radius_m": 0.5,
            "k": null,
            "max_neighbors": null,
            "preparation": {"centering": "center", "scale_normalization": "none"}
          }
        }
      }
    }
  }
}
```

- `geometric_descriptor` não precisa de runtime. `ptv3` precisa de um provider para `point_representation.encoder` (`resources.providers` ou `providers=`) que devolva um `PTv3Runtime`, por exemplo o `PointceptPTv3Runtime` sobre o clone do Pointcept e o checkpoint de hash fixado. O comando precisa rodar no ambiente do Pointcept (venv separado com torch, spconv e torch-scatter); o exemplo completo, o provider e o ambiente estão em [PTv3 pelo runtime](../../point_representation/docs/ptv3.md#pelo-runtime). Sem o provider, o preflight bloqueia em `components.point_representation.encoder`, sem fallback para o descritor.
- O run grava a versão do código: `contextmap run` e `contextmap context run` precisam de `--code-identity` (sem ela, o preflight bloqueia em `stages.point_representation`).
- O exemplo é JSON porque `k` e `max_neighbors` são obrigatórios e nulos numa política de raio, e TOML não tem `null`.
- `contextmap run` roda o estágio quando o plano é completo ou quando ele é um alvo (`--stage point_representation`); um alvo que não depende dele (por exemplo `--stage semantic_fusion`) não o roda. `contextmap context run` o executa como estágio de contexto, e a `ContextRun` grava o run; `contextmap context build` não o fornece nem o recomputa, porque a materialização não o consome.

## Lacunas conhecidas

- **Perfil sem backends.** Escolher SAM3, DINOv3, Qwen/Gemini ou FAST-LIO como canônicos é uma decisão científica que pertence à validação end-to-end, não ao runtime. Até lá, um experimento fornece um arquivo de configuração que seleciona os backends.
- **Parâmetros por capability.** A validação dos parâmetros de cada backend (obrigatórios, tipos, faixas) acontece em `compose()`, instanciando a configuração da própria capability (`build_config()`); a resolução da configuração só garante valores JSON finitos e sem aparência de segredo. Políticas que hoje são apenas parâmetros (sincronização, voxelização, oclusão) ganham ponto de variação quando um executor de estágio as consumir.
- **Perfil global e políticas internas.** `canonical/1` é a topologia global. Descoberta,
  montagem de regiões, extração de features e construção de requests têm políticas de seus
  owners, com identidades próprias. Não existe preset interno monolítico que publique regiões,
  features e claims no mesmo artifact.

- **`context_map`.** `pipeline.stages.context_map` é recusado como estágio desconhecido: `canonical/1` não o declara ainda; é trabalho de outro milestone (End-to-End Validation).
