# Arquitetura de Semantic Interpretation

`semantic_interpretation` é uma capability e um stage de topo. A saída de uma execução é o
`SemanticInterpretationArtifact`; a composição com regiões e features é uma view de leitura,
sem identidade própria de inferência.

## Ownership e dependências

- `region_perception` fornece `Region2D` e um `RegionMaskResolver` público. Esta capability não
  descobre, redefine nem republica regiões ou máscaras upstream.
- `visual_perception` fornece a imagem preparada e os tipos de referências visuais. O stage
  semântico não precisa reexecutar feature extraction.
- `ingestion` fornece `SourceObservationId`, que identifica a observação física.
- `shared` fornece proveniência de backend, tipos JSON e publicação atômica; não possui uma
  confiança genérica.
- Runtime compõe o `SemanticInterpreter`, resolve políticas e seleciona artifacts. O serviço
  desta capability possui a montagem científica de requests, views e condicionamento.

## Execução e substituição

Qwen, Gemini, Eagle-2.5 e Florence-2 implementam `SemanticInterpreter`. O port declara os modos
e evidências aceitos antes de executar. SDKs e runtimes de modelo ficam em `adapters/`, nunca
nos contratos. Não existe fallback implícito de backend.

As instruções dos prompts e os schemas de respostas publicados permanecem preservados. O
header do request identifica agora o run semântico. Essa mudança modifica o fingerprint do
prompt renderizado, pois o fingerprint mede o texto inteiro, sem reescrever o conteúdo das
políticas científicas.

`scene_context` fica no mesmo artifact das claims. Seu condicionamento é acíclico: o request
de cena não recebe contexto, e uma região só recebe o contexto exato da mesma observação e
produtor. Um contexto ausente quando solicitado impede a chamada; não é omitido silenciosamente.

## Artifact

Schema inicial `0.1.0`, identidade imutável do run, sequência e seleção explícitas,
`region_artifact_id`, cobertura de observações, código/configuração efetivos e inventário SHA-256.
Compatibilidade de formatos históricos não faz parte deste novo contrato.

## Composição downstream

Fusão e scoring selecionam explicitamente os artifacts regional, visual e semântico.
O producer semântico não publica embeddings nem incorpora uma identidade de run visual.
`PerceptionResult`, quando usado para reunir esses inputs, é uma view derivada por observação,
com os IDs dos artifacts produtores e sem identidade própria de run ou resultado. Claims e
contexto preservam seu `semantic_interpretation_run_id` ao atravessar essa view.
