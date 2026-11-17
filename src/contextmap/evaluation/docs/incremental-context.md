# Invariância à partição do contexto incremental (#501)

A afirmação central da v0.1.1 é que executar o contexto em várias `ContextRun`s sobre uma fundação espacial fixa não muda o resultado só porque as mesmas observações físicas foram divididas. `contextmap.evaluation.incremental_context` é a comparação objetiva dessa afirmação, com as definições pré-registradas na #501 (comentário de 2026-09-26) antes de qualquer corrida real. O experimento que a usa sobre dados reais está em [`experiments/incremental-partition-invariance-20260927/`](../../../../experiments/incremental-partition-invariance-20260927/README.md); o braço de CI (`tests/end_to_end/test_runtime_chain.py`, `TestPartitionInvariance`) usa a mesma leitura.

Igualdade byte a byte dos artifacts **não** é o critério: ids de `ContextRun`, de run, de suporte, de entidade e de relação são identidades de execução e diferem legitimamente. A comparação troca cada identidade local por conteúdo. E o backend semântico é estocástico sob rerun (29/90 claims iguais no v0.1.0), então a comparação tem duas camadas.

## Leitura de um build: `read_build_state`

`read_build_state(build_run, workspace=...)` lê um `ContextBuild` materializado só pelos leitores públicos (`read_context_build`, `SemanticFusionRunReader`, `SemanticMappingRunReader`, `ContextMapArtifactReader`, `PerceptionRunReader` dos runs de percepção que o build congelou) e devolve um `BuildState`:

| Campo | Conteúdo |
|---|---|
| `counts` (`BuildCounts`) | observações físicas, resultados de inferência, suportes, contribuições, observações excluídas, entidades, candidatos rejeitados, entidades resolvidas, relações e relações por estado, cada uma em separado |
| `layer_one` (`LayerOneState`) | o estado exato, abaixo |
| `layer_two` (`LayerTwoState`) | o estado que depende das claims, abaixo |
| `validation` (`MapValidation`) | a validação `FULL` do `ContextMapArtifact` pelo validador dele: status, erros e avisos por código |
| `integrity_problems` | problemas de inventário dos runs lidos (`verify_integrity()` de fusão, Semantic Mapping e percepção) |

A chave que pareia builds é `GeometryKey`: o conjunto de ids de geometria da fundação que um suporte ou uma entidade cobre. Todo build de uma fundação referencia os mesmos ids, então a chave não depende da execução. Duas entidades de um mesmo build com a mesma geometria são recusadas (`ValueError`): a chave não conseguiria pareá-las.

## Camada 1, exata

`LayerOneState` guarda, ordenados, o `map_id` da fundação, as contagens de observações físicas, resultados de inferência e observações excluídas, a geometria de cada suporte, cada contribuição como `(geometria do suporte, frame, conteúdo da região)` (uma por região × frame × inferência), a geometria de cada entidade, as rejeições por motivo, cada entidade resolvida com a geometria dos seus membros, cada relação como `(predicado, geometria do sujeito, geometria do objeto, estado, tipos de incerteza)` e as capacidades declaradas do `ContextMap`. O conteúdo regional é uma chave de geometria da caixa, dimensões da imagem, área, tipo, aceitação e hash dos pixels de máscara publicado no índice. Assim, a identidade do produtor regional, que muda entre runs independentes, não cria uma divergência científica fictícia; máscaras diferentes ainda são distinguíveis. As regiões de cada resultado de inferência, por frame, entram **só como diagnóstico**, para localizar uma divergência.

Nada de ponto flutuante entra nas dimensões comparadas, então não há tolerância numérica: a comparação é igualdade. `compare_layer_one` compara cada dimensão como multiconjunto (um escalar é uma coleção de um item) e devolve, por dimensão, se é igual, quantos itens só cada lado tem e até 20 exemplos de cada lado, com a geometria como `geometry:<16 hex>x<n>`. `judge_layer_one(w1=, w2=, chunked=)` decide H1:

| `LayerOneOutcome` | Quando |
|---|---|
| `precondition_failed` | `W1 ≠ W2`: a camada 1 não é exata sob rerun; é achado, e reclassificar é decisão nova |
| `partition_divergence` | `W1 = W2` e `C` difere de algum: divergência causada pela partição, bug até prova em contrário |
| `holds` | `C = W1 = W2` |

## Camada 2, estatística

`LayerTwoState` guarda as chaves canônicas de claim de cada frame percebido (`(frame, conteúdo da região ou None, hipótese, papel, categoria, confiança)`), o `AmbiguityState` e o conjunto de labels de cada entidade da Semantic Mapping pela geometria, e as contagens da fusão que dependem das claims (`claims`, `hypotheses`, `evidence_stances`, `abstaining_claims`, `uncertainty`, `supports_with_uncertainty`, `scorer_signals`), que o relatório põe lado a lado sem decidir nada com elas. O registro histórico 29/90 da v0.1.0 mantém a chave e o dataset daquela avaliação; a comparação atual usa conteúdo regional porque os novos IDs têm escopo de produtor.

Cada concordância é dividida em termos por cluster (`ClusterTerms`: concordantes e comparados por cluster):

- `claim_agreement` — `M_claim`: por frame, o tamanho da interseção e o da união das chaves; a soma sobre os frames dá o Jaccard dos 29/90. Um frame que só um lado percebeu conta as claims dele como discordância.
- `paired_agreement` — `M_state` e `M_label`: por entidade, 1 comparado e 1 ou 0 concordante quando os dois lados têm a chave; uma entidade sem par é um cluster com nada a comparar (0/0), porque a falta de par já é divergência da camada 1.

`partition_effect(w1_c, w2_c, w1_w2, resamples=, seed=, confidence=)` estima `D = ½[agr(W1,C) + agr(W2,C)] − agr(W1,W2)`:

- os três pares são alinhados na união dos clusters (um cluster ausente de um par vale 0/0);
- cada reamostragem sorteia, com reposição, tantos clusters quantos há na união, com `numpy.random.default_rng(seed)`; o mesmo sorteio alimenta os três pares (pareado), e cada concordância é a razão das somas sobre os clusters sorteados;
- o IC é o intervalo percentil da confiança pedida; uma reamostragem em que algum par não tem nada a comparar fica fora dele e é contada em `undefined_resamples`, nunca substituída;
- sem nada a comparar num par, `point`, `lower` e `upper` são `None`.

`decide_equivalence(effect, delta=)` aplica a regra na ordem registrada: IC inteiro abaixo de 0 → `rejects_invariance`; senão, limite inferior ≥ −δ → `equivalent_within_delta`; senão `inconclusive`; sem IC, `not_measurable`.

## Cenários 2 e 3

- `judge_repeated_inference(w1=, w2=, both=)` — cenário 2, exato: os três builds contam as mesmas observações físicas e o build sobre as duas execuções conta os resultados de inferência de cada uma (`R = W1 + W2`).
- `stage_dispositions(run)` — como cada estágio de um run obteve seu artifact, lido de `execution.json`: `provided`, `executed`, `reused` ou `recomputed`.
- `judge_downstream_only(baseline=, changed=, baseline_build=, changed_build=)` — cenário 3, estrutural: os dois builds recebem percepção e associação (e representação, quando habilitada) prontas, sem executá-las, e têm identidades diferentes. As saídas podem diferir.

## Relatório

`encode_incremental_context(valor)` dá a forma JSON de qualquer valor deste módulo: dataclasses como objetos, enums pelo valor, `GeometryKey` pela forma curta (também como chave), conjuntos como listas ordenadas. Não resume nem descarta nada.

## Limitações

- As do protocolo: com três execuções, o bootstrap captura a variação entre frames e entidades, mas não a variância da concordância entre execuções (há um único par `W1 × W2`); δ = 0,10 foi escolhido antes da corrida, não derivado; nada aqui mede qualidade semântica.
- As entidades de `M_state` e `M_label` são as da Semantic Mapping, o dono do `AmbiguityState`. As entidades resolvidas do `ContextMap` entram na camada 1, pela geometria e pelos membros.
- Um perfil que ligue canais semânticos ou de aparência na Entity Resolution move resolução e relações para a camada 2; esta leitura continua comparando-as na camada 1, então esse perfil exige outra definição antes de ser comparado.
