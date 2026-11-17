# Suporte da pontuação semântica contra a referência (#530, #658)

A #530 pergunta se a similaridade CLIP/AlphaCLIP traz informação própria para ranquear ou sustentar as hipóteses do VLM. O alvo de qualidade é a referência: "Human/reference correctness remains the quality target; scorer/VLM agreement alone is not correctness". `evaluate_semantic_scoring()` (`contextmap.evaluation.semantic_scoring`) mede isso sobre `SemanticScore` (#652), `SemanticClaim` e `SemanticAnnotation`, sem modelo e sem alterar nenhuma claim. `join_scored_regions()` monta essa entrada a partir dos runs persistidos e do reference set, pela junção de regiões descrita abaixo.

## Entrada

- `ScoredRegion`: as claims de **uma** região do run de percepção (primária e alternativas, pontuadas ou não), os `SemanticScore` dessas claims e a `SemanticAnnotation` da região de referência juntada a essa região do run (`None` quando não há anotação). Uma claim de cena é recusada: o scoring de cena é N/A desde a #652 (D2).
- `SemanticScoringEvaluationContext`: avaliação, versão do reference set, seleção, run de percepção pinado, runs de scoring, os schemas de anotação que o reference set fornece e `region_join_policy_id`, a política versionada que juntou regiões de referência a regiões do run.
- `SemanticScoringEvaluationPolicy`: `policy_id` e `minimum_class_count`, o mínimo declarado de positivos **e** de negativos pontuados para o AUROC e o AUPRC agrupados. Não há default: o valor é fixado pelo protocolo da #530 antes de qualquer braço rodar, e o relatório cita a política.
- `region_joins` (opcional): as junções que anexaram as anotações, para o relatório contar o que elas casaram e deixaram sem par (`region_join`). Uma junção de outra política que não a do contexto é recusada.

A anotação chega pela ponte `SemanticAnnotationSet.to_semantic_annotation()`, que entrega `rejected_concepts` como `rejected_hypotheses` e um registro `UNKNOWN` como `abstention_expected=True` (#653, lacuna 1; ver [`annotations.md`](annotations.md)).

## Rótulo de cada score

Sob `casefold-exact/1` (`MATCHING_POLICY`), a hipótese de uma claim pontuada é:

| Rótulo | Quando |
|---|---|
| positiva | casa com uma hipótese aceitável da anotação |
| negativa | casa com uma hipótese rejeitada, a única verdade negativa |
| não avaliada | nenhuma das duas, ou a região não tem anotação |

Não avaliada nunca conta como negativa (#197: "Missing/unannotated open-vocabulary concepts remain N/A"). Um registro `UNKNOWN` não tem positivas; as rejeitadas que ele tiver continuam negativas.

## Regras

- Só `value` é lido, na escala de `score_type`: o cosseno fica em `[-1, 1]`, sem remapeamento, e nunca é tratado como probabilidade. `calibrated_probability` nunca é lido: nenhuma calibração com `CalibrationSplitPair` existe ([`score-calibration.md`](score-calibration.md)).
- Toda métrica é por scorer (proveniência completa, com o fingerprint da configuração), espaço de embedding e tipo de score. Dois espaços são dois blocos, nunca agrupados, e não existe score combinado.
- Uma claim sem score fica fora de todo denominador; ela aparece só em `unscored_claim_count`. Uma claim com dois scores do mesmo scorer no mesmo espaço é recusada.
- Repetições não são amostras novas: só `repeat_index == 0` entra na qualidade e no diagnóstico; `repeat_region_count` conta as outras. A mesma região duas vezes na mesma repetição é recusada.

## Métricas de qualidade

Estágio `semantic_scoring` do registro (versão 4), separado de `semantic_interpretation`, como pede a #530 ("Results remain a separate evidence channel"). Todas exigem `contextmap.reference.semantics/v1` e excluem o que não está anotado.

| Métrica | Definição | População |
|---|---|---|
| `scoring.ranking.top1_accuracy` | regiões cuja claim rotulada de maior score é positiva ÷ regiões elegíveis | regiões com ≥ 1 positiva e ≥ 1 negativa pontuadas |
| `scoring.ranking.pairwise_accuracy` | média, entre as regiões elegíveis, dos pares (positiva, negativa) da região ordenados corretamente, um empate valendo 1/2 | as mesmas regiões |
| `scoring.auroc` | pares (positiva, negativa) de todas as regiões ordenados corretamente, mais metade dos empates, ÷ todos os pares | positivas e negativas pontuadas, com ≥ `minimum_class_count` de cada classe |
| `scoring.auprc` | precisão média das positivas ordenadas por score decrescente: soma, por limiar distinto, do recall ganho vezes a precisão no limiar; scores empatados formam um único limiar | a mesma do AUROC |

Empates são declarados, nunca desfeitos em silêncio:

- no top-1, um empate no topo entre uma positiva e uma negativa entra no denominador e nunca como acerto; `top1_tie_count` os conta;
- nos pares, o empate vale 1/2 (convenção de Mann-Whitney: um scorer constante vale exatamente o acaso); `pairwise_tie_count` os conta.

`pairwise_accuracy` é, por construção, o AUROC por região com média macro. O relatório traz as duas definições de ranking (D2 da #658: top-1 e por pares) e as duas agregações de AUROC (D3: agrupada, `scoring.auroc`; macro por região, `scoring.ranking.pairwise_accuracy`).

**Métrica principal da #530: `scoring.ranking.pairwise_accuracy`** (decisão registrada na #658). Ela compara só hipóteses anotadas da mesma região, então não depende de o cosseno ser comparável entre regiões, o que ninguém garante: o valor muda com o conteúdo do recorte e com o texto. O AUROC e o AUPRC agrupados são evidência secundária: dizem se um único limiar global separaria aceitáveis de rejeitadas, a pergunta de quem quisesse consumir o cosseno bruto com um corte fixo. O top-1 também é secundário, mais grosso: uma região vale um acerto ou um erro, qualquer que seja o número de alternativas. O AUPRC depende da prevalência de positivas; leia-o com `positive_count` e `negative_count` do mesmo bloco.

Cada bloco de qualidade também traz as distribuições de `value` das positivas e das negativas (contagem, mínimo, quartis e máximo pelo percentil `nearest-rank`, o mesmo `PERCENTILE_METHOD` da Semantic Interpretation) e `samples`, cada score pontuado da execução primária com `score_id`, `claim_id`, observação, região, hipótese, papel, rótulo e valor.

### N/A, sempre com o motivo

Uma métrica sem valor é um `SupportMetric` com `value=None`, a população encontrada e `not_applicable_reason`. Nunca vira zero.

| Situação | Métricas | Motivo |
|---|---|---|
| o reference set não tem a família `semantics` (hoje) | as quatro | `the reference set provides no contextmap.reference.semantics/v1 annotations` |
| nenhuma região tem positiva e negativa pontuadas | ranking | `no region of the primary run has both a positive and a negative scored claim` |
| menos de `minimum_class_count` positivas ou negativas | AUROC e AUPRC | o mínimo e as contagens encontradas |

Uma região com anotação exige a família `semantics` no contexto e um `region_join_policy_id`; sem eles a avaliação é recusada (`SemanticScoringEvaluationError`), porque a origem da anotação ficaria sem registro.

## Diagnóstico sem referência

`ScorerDiagnostics`, um bloco por scorer e espaço, fica fora da qualidade:

- a distribuição de todos os scores da execução primária, rotulados ou não;
- `primary_agreement`: entre as regiões com a primária do VLM pontuada e ao menos outra claim pontuada, a fração em que a primária tem o maior score, estritamente. Um empate no topo entra no denominador e nunca como concordância (`primary_agreement_tie_count`). Concordar com o VLM não é estar certo.

## Envelope comum

`semantic_scoring_evaluation_report()` leva o relatório ao envelope de [`metrics.md`](metrics.md): um resultado por métrica, scorer e espaço, com os estratos `scorer` (`backend/modelo/versão/fingerprint`, de `scorer_label()`) e `embedding_space_id`. Uma métrica sem valor é `not_applicable`; o motivo segue no `stage_report` (`encode_semantic_scoring_report()`). Sem nenhum score, cada métrica sai como um único `not_applicable` sem estrato. Os artifacts de entrada são o run de percepção e cada run de scoring. O diagnóstico não vira métrica, e nenhuma métrica de performance é levantada.

## Junção região de referência → região do run

A anotação vem de uma região da **referência** (família `regions`), e as claims usam o `region_id` local do run de percepção. `contextmap.evaluation.region_join` junta as duas (#653, lacuna 2), e a mesma política precisa servir a todo braço comparado:

- `RegionJoinPolicy(policy_id, min_iou)`: a versão da regra e o IoU mínimo, em `(0, 1]`, sem default.
- A sobreposição é a regra de IoU do Region Discovery (`region.iou.mean`), reaproveitada do dono (`region_pixels`, `box_pixels`, `mask_pixels`, `pixel_set_iou` em `region_discovery`): conjuntos de pixels, uma máscara dando o seu primeiro plano e uma caixa os pixels cujo centro cai dentro dela, nos dois lados.
- `join_frame_regions()` é gulosa e um-para-um: todo par com IoU ≥ `min_iou` é tomado por IoU decrescente, com empate desfeito pelo id da região de referência e depois pelo da região do run, e um par é pulado quando um dos lados já tem par.
- Uma região do run sem par não recebe expectativa (`annotation=None`); uma região de referência sem par é listada no `FrameRegionJoin` e contada, nunca descartada. `summarize_region_joins()` dá o `RegionJoinSummary` (observações, pares, regiões de referência e do run sem par) e recusa junções de mais de uma política.
- Uma região do run persistida com `mask_reference` precisa trazer a máscara carregada: ler a caixa no lugar dela mudaria o IoU sem aviso. Uma região de outra observação ou de outro tamanho de imagem é recusada.

`join_scored_regions(perception=..., scoring=..., regions=..., semantics=..., policy=...)` lê o run de percepção pinado e os runs de scoring pelos leitores públicos e devolve `JoinedScoredRegions`:

- junta toda observação do run que tem anotação de regiões, com as máscaras carregadas do mask store do run;
- cada região do run com claims de região vira um `ScoredRegion` com as claims, os scores de todos os runs de scoring e a anotação semântica da região de referência juntada; sem par, sem anotação de regiões na observação, sem registro semântico ou sem a família `semantics`, a anotação é `None`;
- recusa um run de scoring de outro run de percepção, uma observação anotada sob duas amostras e um score que nenhuma claim de região do run possui;
- `JoinedScoredRegions.context()` preenche o contexto com a política de junção, os runs e os schemas disponíveis, e `joins` vai para `evaluate_semantic_scoring(region_joins=...)`, que reporta o `RegionJoinSummary` em `region_join`.

A escolha dos arquivos do reference set (qual `trust`, qual anotador) fica com quem chama: `regions` e `semantics` chegam como os conjuntos de anotação já lidos.

## O que fica N/A antes do freeze

| Métrica da #530 | Estado |
|---|---|
| distribuições para aceitáveis e rejeitadas, ranking, AUROC/AUPRC | implementadas, com a junção de regiões; N/A enquanto o reference set não tiver a família `semantics` (#653) |
| sensibilidade à view | N/A: um run de percepção tem um único `region_features`, então uma claim nunca é pontuada contra features de duas views (#652) |
| latência e VRAM do scorer | N/A: o `SemanticScoringRunArtifact` não registra tempo nem memória |
| suporte a claims de cena | N/A: nenhum slot do runtime produz a feature CLIP global (#652, D2) |
| efeito até o mapa final (#180) | N/A: nenhuma política de fusão consome `SemanticScore` (#652, problema 4) |

## Seleção dos produtores

`join_scored_regions` recebe leitores regionais e semânticos explícitos. Valida ambos contra a
linhagem dos runs de scoring e exige que compartilhem um produtor visual. Máscaras vêm do
artifact regional, claims do semântico e scores dos runs selecionados. O contexto e o envelope
do relatório registram `region_artifact_id`, `visual_evidence_artifact_id` e
`semantic_interpretation_artifact_id` individualmente. O join usa observação física e claim;
um resultado composto não cria uma nova identidade.
