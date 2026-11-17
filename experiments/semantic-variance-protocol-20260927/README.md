# Protocolo de variância para backends semânticos não determinísticos (issue #580)

Este experimento separa o mecanismo da divergência semântica medida no run0005 e define como a
variância entre repetições é reportada. Ele cobre EV-01, EV-09 e EV-10 da auditoria 2026-09.

> **Estado: preparado, ainda não executado.** O harness e a análise estão prontos e testados
> (`tests/evaluation/test_semantic_variance_protocol.py` e
> `tests/evaluation/test_semantic_variance_protocol_harness.py`), mas este diretório **não** tem
> resultado. O experimento exige GPU, os checkpoints e o `SequenceArtifact` real do corridor-02,
> que não estavam no ambiente em que foi preparado. Não há número novo a citar até o comando
> abaixo rodar e o resultado ser registrado aqui.

## Baseline

- **run0005** ([`acceptance-report-run0005.md`](../e2e-real-canonical-run-20260923/acceptance-report-run0005.md)):
  duas execuções reais e independentes da mesma configuração canônica de Visual Perception
  (SAM2.1-hiera-tiny + DINOv2-base + CLIP-ViT-L/14 + Qwen3-VL-4B-Instruct nf4, gulosa,
  `temperature=0.0`) sobre os mesmos 20 frames do corridor-02 concordaram em **29/90 claims
  canônicas (32,2%, medido)**. Region discovery: 0/20 frames divergentes. Execuções semânticas
  bem-sucedidas: 76/120 e 78/120.
- Gate `reproducibility.rerun_equivalence` do cenário 1.0.4: **FAILED**, e continua assim.
- O baseline tem **um único par**: a distribuição da concordância é desconhecida, e um par não
  distingue variância alta de azar.
- Os 32,2% continuam registrados como resultado negativo medido do run0005. Nenhum resultado deste
  experimento os reinterpreta retroativamente.

## Hipóteses (da issue)

- **H1 (EV-01)**: a divergência vem de não-determinismo de execução em GPU cascateando pela
  geração autorregressiva. Com as flags de determinismo do torch (seeds fixas,
  `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG`), no mesmo hardware e
  driver, a concordância exata entre repetições sobe para 100%.
- **H2 (EV-10)**: decompondo a igualdade por componente, `confidence` é a componente mais frágil,
  e `hypothesis` + `role` + `category` concordam substancialmente mais do que a tupla completa.
- **H3 (EV-09, lacuna verificada)**: o aparato de avaliação não expressa variância entre
  repetições. `repetitions_per_sample` (`src/contextmap/evaluation/experiments.py`) existe sem
  relatório de variância, não há intervalo de confiança nem bootstrap em
  `src/contextmap/evaluation`, e `evaluator_reproducibility.py` cobre o avaliador, não o backend.
  H3 não é testada por dado: este experimento produz, fora de `src/`, o relatório min/mediana/máx
  que falta, e a mudança de protocolo em `src/contextmap/evaluation` depende de H1 e H2.

### Notas de pré-registro (verificáveis no código, não são resultado)

1. **`confidence` é nula por construção.** O Qwen interpreta sob
   `SemanticConfidencePolicy.UNSCORED_ONLY`, e sob essa política o parser recusa qualquer
   `confidence` não nula (`claim[i].confidence must be null`, em
   `src/contextmap/visual_perception/semantic_prompt.py`). Isso já valia no commit do run0005
   (`ee3021a`). Toda claim canônica comparada, lá e aqui, tem `confidence = None`. Então a
   projeção `confidence` só mede presença, e a tupla completa coincide com
   `hypothesis+role+category`. A premissa de H2 ("float produzido pelo LLM") não se aplica à
   configuração canônica. O relatório mostra isso nos dados (`claims_with_confidence`), e a
   decomposição por `hypothesis`, `role`, `category` e `slot` continua informativa. Reformular H2
   é decisão do mantenedor, antes ou depois do run; o harness mede todas as projeções de qualquer
   jeito.
2. **O braço A é a configuração canônica *atual*, como a issue pede, não a do run0005.** Desde
   `ee3021a`, o texto dos prompts `scene/v1` e `region/v1` mudou (`40b1820`: "omit it or set it to
   null") e o parser aceita a chave `confidence` ausente. A identidade do run de percepção também
   passou a incluir a configuração efetiva do discovery (#636), então o `configuration_digest`
   não é comparável com o do run0005. O que se compara com o baseline é a **forma** da
   distribuição, não um par contra um par.
3. **Todas as repetições têm o mesmo `run_id`.** O runtime deriva a identidade de estágio,
   configuração e hashes de entrada (`StageRequest.identity()`), e isso pressupõe determinismo.
   As repetições se distinguem pelo diretório de run e pelo `content_hash`, que diverge
   exatamente quando o conteúdo diverge. O harness nunca passa política de reuso, então cada
   repetição é uma execução real. A consequência para o reuso da runtime com um backend não
   determinístico está fora do escopo daqui.
4. **Ameaças à validade registradas, não controladas:** `PYTHONHASHSEED` não é fixado em nenhum
   braço (o valor e `sys.flags.hash_randomization` ficam no registro); os kernels do bitsandbytes
   (nf4) não são governados pelas flags do torch; e `use_deterministic_algorithms(True)` faz o
   torch levantar erro em operação sem implementação determinística. Nesse último caso o braço B
   **falha** e o registro guarda o erro. Não existe `warn_only`: rodar com ele seria outro braço.

## Braços

| Braço | O que roda | Variável |
|---|---|---|
| **A** | a configuração canônica de Visual Perception, como está | — |
| **B** | A + seeds fixas (Python, NumPy, torch, CUDA), `torch.use_deterministic_algorithms(True)` e `CUBLAS_WORKSPACE_CONFIG=:4096:8` | só as flags |

- **n ≥ 3 repetições por braço** (padrão 3), cada uma um **processo novo**: a divergência do
  run0005 foi medida entre execuções independentes. As 2 repetições por request da validação de
  2026-09-21 (`src/contextmap/evaluation/docs/semantic-interpretation.md`) devolveram texto
  idêntico, mas nada registra se rodaram em processos separados. Repetir dentro de um processo
  mediria outra coisa.
- **Intercaladas** (A1, B1, A2, B2, …), para que uma deriva da máquina no tempo não caia num braço
  só. Mesma máquina, mesmo driver, mesma configuração em todas.
- `CUBLAS_WORKSPACE_CONFIG` precisa existir antes de o CUDA iniciar, então entra na linha de
  comando do processo do braço B (`env CUBLAS_WORKSPACE_CONFIG=:4096:8 …`). O braço A roda com
  `env -u CUBLAS_WORKSPACE_CONFIG`, mesmo que o shell a exporte, e cada repetição recusa começar
  se o ambiente contradiz o seu braço. As seeds e `use_deterministic_algorithms` são aplicados
  antes da composição, que é quem carrega os modelos.
- A etapa executada é o estágio `visual_perception` inteiro, pelo ponto de entrada público da
  runtime (`contextmap.runtime.Runtime.run`, a mesma composição de `contextmap stage
  visual_perception`), como no baseline: SAM2, DINOv2, CLIP e Qwen rodam a cada repetição.

## Dataset

Os **20 frames reais do corridor-02 congelados no cenário 1.0.4**, lidos do snapshot
[`solution-1-canonical-1.0.4-corridor-02-sample.json`](../../src/contextmap/evaluation/docs/scenarios/solution-1-canonical-1.0.4-corridor-02-sample.json)
(digest `sha256:90bf0e26107d13ed8831a7acd795cee6ebb00bc2d5f01dd30819b042eca791e3`), do
`SequenceArtifact` `720a486de8d44c16a9d3d2ff9fa7b1a4`, que é o único aceito. O cenário 1.0.4 não
muda. A 1.0.5 mantém a mesma amostra, e um teste confere isso.

O estágio processa toda imagem da sequência de entrada. Por isso o harness copia as 20 imagens,
sem alteração, para um `SequenceArtifact` derivado dentro do workspace do experimento. A
identidade dele depende só do artifact de origem e dos frames. A provenance nomeia o artifact de
origem, o `selection_id` (`ExplicitIdsSelection`) e o snapshot 1.0.4, e a calibração é copiada.
Cada frame é conferido contra o id **e** o timestamp congelados. Não há reference set: mede-se só
reprodutibilidade, não qualidade.

## Métricas

Todas por **par de repetições** (os C(n, 2) pares de cada braço) e resumidas como
**mín / mediana / máx sobre os pares do braço**. Com um número par de pares, a mediana é a média
dos dois centrais.

- **Concordância exata da tupla completa.** A canonicalização é a mesma do
  [`15_visual_perception_content_equivalence.py`](../e2e-real-canonical-run-20260923/scripts-run5/15_visual_perception_content_equivalence.py):
  cada claim é `(source_observation_id, region_id, hypothesis, role, category, confidence)`, as
  claims de um run num frame formam um conjunto, e a concordância é
  `iguais / (iguais + só_no_primeiro + só_no_segundo)`, somada sobre os frames presentes nos dois
  runs. Um teste confere que a chave é a do script 15. Duas diferenças de relato, nenhuma na
  tupla: `0/0` é **indefinido** (o script 15 dava 1,0), e frame ausente de um lado é contado à
  parte de frame com regiões divergentes (o script 15 somava os dois).
- **Concordância por componente.** A mesma conta sobre a projeção das claims em `hypothesis`,
  `role`, `category` e `confidence`, em `hypothesis+role+category` (a comparação de H2) e em
  `slot` (só o frame e a região que receberam claim). Toda projeção mantém a chave
  `(source_observation_id, region_id)`.
- **Execuções semânticas bem-sucedidas por run**: `attempted`, `parsed` e `parse_failed` de
  `account_semantic_attempts` (os dois streams contratuais do artifact), com mín/mediana/máx sobre
  os runs do braço.
- **Validade do pareamento**: regiões divergentes e frames ausentes por par. As claims são
  pareadas por `region_id`, que só é comparável entre runs cujas regiões coincidem, como no
  run0005.

Os runs de um braço precisam ter o mesmo `configuration_digest` de pipeline, senão não são
repetições, e a análise recusa. Os dois braços também devem ter o mesmo, porque as flags não são
configuração, e o relatório diz se têm (`arms_share_configuration`).

## Critérios e regra de decisão (da issue)

**Critérios.** H1 é corroborada se o braço B atinge 100% de concordância exata entre suas
repetições, e rejeitada se qualquer par divergir (`hypotheses.H1.verdict`: `corroborated`,
`rejected` ou `undetermined` sem nenhuma claim). H2 é corroborada se a concordância de
`hypothesis+role+category` exceder materialmente a da tupla completa no braço A. A issue não
quantifica "materialmente", então o relatório dá as medianas e a diferença, sem veredito
automático (e veja a nota 1).

**Regra de decisão.**

- **H1 corroborada** → issue feat: incorporar as flags de determinismo à configuração canônica
  versionada de Visual Perception (entram no fingerprint/digest). O gate segue exigível como
  igualdade exata.
- **H1 rejeitada** (o backend continua não determinístico) → issues feat derivadas:
  (a) protocolo de variância: um backend que declara não-determinismo exige n repetições e
  relatório mín/mediana/máx por métrica (hoje `repetitions_per_sample` existe sem relatório de
  variância, em `experiments.py`); (b) recusar a comparação de braço semântico com 1 repetição
  quando o backend declara não-determinismo; (c) equivalência de claims decomposta por
  componente, com versão tolerante do critério numa **nova** versão de cenário. A 1.0.4 é
  congelada por digest: a mudança exige novo `SCENARIO_VERSION` (`end_to_end.py`), nunca
  redefinição silenciosa. A escolha entre tolerância e determinismo usa o resultado de H2.
- Em qualquer resultado: 32,2% continua registrado como resultado negativo medido do run0005,
  sem reinterpretação retroativa.

## Proveniência registrada

| O quê | Onde |
|---|---|
| commit e working tree suja ou não | `experiment.json` → `code`; cada run da runtime → `code_identity` |
| digest da configuração efetiva | cada repetição → `config_digest` (e o `effective_config.json` do próprio run); `configuration_digest` de pipeline por run no relatório; SHA-256 das duas camadas de configuração |
| hardware e driver | cada repetição → `hardware` (`nvidia-smi`: GPU, driver, memória, barramento; torch, CUDA, cuDNN, capability, flags do cuDNN), `environment` do run e `packages` (torch, transformers, bitsandbytes, sam2, …) |
| seeds e flags | cada repetição → `determinism`, `environment_variables`, `python_hash_randomization` |
| ids dos artifacts | sequência de origem e derivada (`sequence`), e por repetição o `artifact_id`, o `content_hash` e a localização do `PerceptionRunArtifact` |
| tempo e memória | cada repetição → `measurement` (`measure_child_process`, o mesmo método do #181) |

Nenhum registro leva caminho absoluto da máquina: as localizações são relativas ao `outputs/`.

## Como executar (máquina com GPU)

**Pré-requisitos**

- O checkout **principal** do repositório no commit a medir. Os braços ficam retidos no
  `outputs/` dele; um *worktree* some com o próprio `outputs/`.
- Um ambiente com `contextmap` e o extra `vision`, mais o SAM 2 (`facebookresearch/sam2`) e o
  `bitsandbytes`, com os checkpoints no cache local.
- O `SequenceArtifact` `720a486de8d44c16a9d3d2ff9fa7b1a4`
  (`outputs/ingest-real/sequences/corridor-02/720a486de8d44c16a9d3d2ff9fa7b1a4`).
- Um arquivo de parâmetros: uma camada de configuração da runtime com os parâmetros canônicos dos
  quatro backends e os alvos `resources.providers` do SAM2 e do Qwen, que não têm loader
  empacotado. É a mesma configuração dos runs `run-0001`/`run-0002` do run0005. O repositório não
  a versiona, e este experimento não a inventa. A seleção dos backends vem do cenário e não deve
  ser repetida. A forma é esta (os valores entre `<…>` são do mantenedor):

  ```json
  {
    "resources": {
      "device": "cuda",
      "providers": {
        "visual_perception.region_discovery": "<módulo>:<loader do SAM2>",
        "visual_perception.semantic_interpretation": "<módulo>:<loader do Qwen>"
      }
    },
    "components": {
      "visual_perception": {
        "region_discovery": {"sam2": {"checkpoint": "<…>", "model_version": "<…>"}},
        "dense_features": {"dinov2": {"checkpoint": "facebook/dinov2-base", "revision": "<sha>"}},
        "region_features": {"clip": {"checkpoint": "openai/clip-vit-large-patch14", "revision": "<sha>"}},
        "semantic_interpretation": {
          "qwen": {
            "model": "Qwen/Qwen3-VL-4B-Instruct", "revision": "<sha>", "quantization": "4bit",
            "temperature": 0.0, "precision": "<…>", "max_new_tokens": <…>,
            "prompt_policy": {"scene": "scene/v1", "region": "region/v1"},
            "view_policy": {"region_views": ["tight_crop"]}
          }
        }
      }
    }
  }
  ```

  `prompt_policy` e `view_policy` não são valores do mantenedor: desde a ablação de percepção
  (milestone 22) a política de request semântico não tem padrão (#542, #524), e a que reproduz o
  request do run0005, anterior a ela, é essa, os prompts `scene/v1`/`region/v1` e uma única view,
  o recorte justo da região ([composição do runtime](../../src/contextmap/runtime/docs/composition.md#política-de-prompt-semântico-542)).
  Sem os dois grupos, o preflight recusa a composição do Qwen.

  Os valores efetivos do run0005 estão nos próprios artifacts dele: cada execução semântica grava
  `effective_configuration` (model, revision, device, precision, max_new_tokens, temperature,
  quantization), e cada região grava o `configuration_fingerprint` do SAM2.
- O módulo dos loaders importável pelos processos das repetições (instalado ou no `PYTHONPATH`,
  que eles herdam). Um `RuntimeProvider` recebe só `(config, secrets)` e devolve só o runtime do
  modelo carregado: o do SAM2 é `Sam2AutomaticMaskRuntime.from_model(model=..., config=config)` e
  o do Qwen é `HuggingFaceQwenRuntime(config=config)`, sem `image_loader` nem `view_root`. O
  `VisualPerceptionExecutor` cria o diretório de rascunho das imagens preparadas e das views
  semânticas dentro de `execute()`, e os adapters do run resolvem as referências nele: o runtime
  do SAM2 recebe os pixels de cada pass, e o do Qwen, os bytes de cada view verificados pelo
  SHA-256 ([o que um provider devolve](../../src/contextmap/runtime/docs/composition.md#o-que-um-provider-devolve-661),
  #661).

**Comando**, da raiz do checkout:

```bash
python experiments/semantic-variance-protocol-20260927/scripts/run_experiment.py \
    --source-sequence outputs/ingest-real/sequences/corridor-02/720a486de8d44c16a9d3d2ff9fa7b1a4 \
    --parameters <arquivo de parâmetros>.json
```

Opções: `--repetitions N` (mínimo 3, padrão 3), `--run-number N` (padrão 1; uma rodada nova usa
outro número), `--seed S` (padrão 0, só o braço B usa), `--outputs-root DIR` (padrão: o
`outputs/` do checkout) e `--preflight-only`.

**Rode primeiro com `--preflight-only`.** Ele não carrega modelo nem escreve nada e confere:

- que a configuração resolve e seleciona os backends do cenário 1.0.4;
- que cada parâmetro passa pela configuração da própria capability;
- que cada alvo de `resources.providers` importa;
- que os módulos opcionais existem;
- que a sequência de origem é a `720a486…`.

Depois, o comando sem a opção:

1. repete o preflight;
2. deriva a sequência dos 20 frames;
3. roda as 2n repetições intercaladas, cada uma medida em processo novo;
4. escreve o relatório.

A execução é **fail-fast**. Uma repetição que sai com status diferente de zero, não deixa registro
ou cujo run não concluiu interrompe o experimento. Nesse caso nada roda depois, `experiment.json`
sai com `"status": "failed"` e a lista `failures`, não há relatório e o comando termina com 1. Um
diretório de rodada que já existe é recusado. Só `"status": "complete"` com saída 0 é evidência.

**Verificação de paridade com o baseline** (recomendada, sem GPU): a análise sobre o próprio par
do run0005 deve reproduzir 29/90 na tupla, com 0 regiões divergentes e 0 frames ausentes.

```bash
python experiments/semantic-variance-protocol-20260927/scripts/semantic_variance.py \
    --arm run0005=outputs/e2e-real/visual_perception/workspace/corridor-02/run-0001/visual_perception,outputs/e2e-real/visual_perception/workspace/corridor-02/run-0002/visual_perception \
    --output outputs/semantic-variance-protocol-20260927/parity-run0005
```

Se não der 29/90, a canonicalização não é a do baseline: pare e reporte.

## Saídas e o que volta para o repositório

```text
outputs/semantic-variance-protocol-20260927/run-0001/
├── config/                          # as duas camadas de configuração usadas
├── workspace/
│   ├── inputs/sequence/             # SequenceArtifact derivado (20 frames)
│   └── corridor-02/run-000k/        # um run da runtime por repetição (journal + visual_perception/)
├── repetitions/<braço>-<k>.json     # registro de cada repetição; <braço>-<k>.log é só diagnóstico
├── experiment.json                  # status, falhas e proveniência
└── report/report.json, report.md    # só quando completo
```

Voltam para `experiments/semantic-variance-protocol-20260927/results/`, para commit:
`experiment.json`, `report/report.json`, `report/report.md` e, se rodada, a verificação de
paridade (`parity-run0005/report.json`). Depois vem uma seção "Resultado" neste README, com os
números e a aplicação da regra de decisão. Os artifacts ficam retidos em `outputs/`, fora do Git.

## Scripts

| Script | Faz |
|---|---|
| `scripts/run_experiment.py` | preflight, sequência derivada, repetições intercaladas e relatório; o comando único |
| `scripts/repetition.py` | uma repetição de um braço, em processo próprio: flags do braço, `Runtime.run` do estágio `visual_perception` e o registro |
| `scripts/semantic_variance.py` | a análise determinística (pareamento, canonicalização, projeções, mín/mediana/máx, H1 e H2); também roda sozinha sobre quaisquer runs |

## Fora de escopo (da issue)

- Fazer o gate "passar" aqui: o experimento isola o mecanismo e define o protocolo, e não altera o
  cenário 1.0.4.
- Implementar a versão tolerante de equivalência ou o protocolo de variância em
  `src/contextmap/evaluation` antes da regra de decisão.
- Reference set real e os 5 gates `blocked` (EV-07/EV-12), zeros implícitos em métricas
  (EV-02/EV-03/EV-08) e a reprodutibilidade do avaliador.
- Otimização de performance e backends fora da configuração canônica.
