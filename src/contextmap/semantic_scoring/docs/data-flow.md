# Fluxo de dados de Semantic Scoring

```text
SequenceArtifact
    -> RegionPerceptionRunArtifact       regiões e máscaras finais
       -> VisualEvidenceArtifact         features visuais
       -> SemanticInterpretationArtifact claims e contexto de cena
    -> seleção explícita dos três produtores compatíveis
    -> view derivada por source_observation_id
    -> SemanticScorer.score(claims, features)
    -> SemanticScore
    -> SemanticScoringRunArtifact
```

O stage semântico e o visual consomem o produtor regional selecionado de forma independente.
Uma reexecução semântica pode usar o mesmo artifact visual. A view não cria um run que una
suas identidades nem exige repetir a extração das features.

## Pareamento

O scorer filtra espaço de embedding, escopo e ancoragem, renderiza a hipótese no template,
codifica o texto e calcula cosseno contra o embedding visual. Um claim sem feature compatível
não gera score. Espaço divergente, normalização inválida ou seleção ambígua são erros de
contrato; não são convertidos em valor zero.

Cada score registra os IDs do claim e da feature, a observação física, a região e o espaço.
O claim preserva seu produtor semântico; a feature preserva seu produtor visual. A seleção
regional e as identidades dos três artifacts ficam na linhagem do scoring.

## Escrita e publicação

`SemanticScoringRunWriter` recebe linhagem, identidade/proveniência do scorer, espaço de
embedding e template. Em `add_result(view, scores)`, valida sequência e produtores,
observação única, referências, espaço, scorer e pareamento regional. Os scores persistidos
recebem IDs qualificados pelo run de scoring.

`finalize()` grava em staging e publica por rename atômico:

```text
<run_dir>/
├── manifest.json
├── outputs/scores.jsonl
└── README.md
```

| Campo do manifest | Conteúdo |
| --- | --- |
| `run_id`, `run_index` | Identidade da execução de scoring |
| `lineage` | Sequência, seleção e artifacts regional, visual e semântico |
| `scorer` | Backend, modelo, versão e fingerprint da configuração |
| `embedding_space`, `embedding_space_id` | Contrato tipado e fingerprint do espaço |
| `text_template` | Template exato usado para as hipóteses |
| `scope`, `score_type`, `score_semantics` | Região, cosseno e declaração da faixa `[-1, 1]` |
| `counts` | Observações processadas, claims regionais/de cena, claims pontuados/não pontuados e scores |
| `file_inventory` | Tamanhos e hashes dos arquivos contratuais |
| `schema_version` | `0.2.0` |

`counts.result_count` conta observações processadas; o nome do campo não introduz um ID de
resultado. `unscored_region_claim_count` distingue claim sem score de score com valor zero.
O reader restaura contagens e inventário publicados no manifest tipado.

## Leitura e junção

`SemanticScoringRunReader.read_scores()` e `iter_scores()` decodificam `SemanticScore`.
`read_manifest()` descreve a execução e sua linhagem; `verify_integrity()` verifica os
payloads locais. Ler esse artifact não carrega os upstreams.

Consumidores como avaliação e fusão verificam a seleção upstream e juntam por produtor,
`source_observation_id`, claim e região. A avaliação de scoring recebe os readers regional,
semântico e os runs de scoring e mantém o produtor visual explícito na linhagem. Scores de
espaços incompatíveis permanecem separados.
