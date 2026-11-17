# Testes e avaliação de Semantic Scoring

## Cobertura existente

| Responsabilidade | Suíte |
| --- | --- |
| Contratos, pareamento, ausência de score e preservação das entradas | `tests/visual_perception/test_semantic_scoring.py` |
| Manifest tipado, contagens, codecs e validação da persistência | `tests/visual_perception/test_semantic_scoring_run.py` |
| Publicação atômica, hashes, imutabilidade e falhas de escrita | `tests/semantic_scoring/test_artifacts_atomic_persistence.py` |
| Três produtores independentes e ancoragem física dos scores | `tests/semantic_scoring/test_independent_producers.py` |
| Torre de texto e templates CLIP/AlphaCLIP | `tests/visual_perception/backends/test_clip_text.py`, `test_alphaclip_text.py` |
| Composição, lineage e execução opcional do stage | `tests/runtime/test_runtime_semantic_scoring.py` |
| Consumo pela fusão | `tests/semantic_fusion/` |
| Junção dos artifacts e métricas | `tests/evaluation/test_semantic_scoring_runs.py`, `test_semantic_scoring_evaluation.py` |
| Imports públicos e fronteiras do núcleo/adapters | `tests/architecture/test_boundaries.py` |

Algumas suítes conservam seu diretório de testes anterior. Seus imports usam os owners atuais;
a localização do arquivo de teste não transfere ownership para visual perception.

## Invariantes protegidos

1. Cosseno permanece em `[-1, 1]`, sem probabilidade implícita.
2. Ausência de feature compatível produz ausência de score, não score zero.
3. Claims e features permanecem intactos.
4. Scores respeitam observação, região, espaço e scorer declarados.
5. Uma view de outro produtor selecionado ou uma observação repetida é recusada.
6. Contagens registram cobertura; leitura tipada preserva o inventário publicado.
7. Artifact finalizado é imutável e uma falha não publica diretório parcial.
8. O stage de scoring mantém separados os três artifacts upstream.

```bash
pytest -q tests/semantic_scoring tests/visual_perception/test_semantic_scoring.py \
  tests/visual_perception/test_semantic_scoring_run.py \
  tests/runtime/test_runtime_semantic_scoring.py -p no:cacheprovider \
  --basetemp="$HOME/.cache/cm2/semantic-scoring-tests"
```

Esses testes verificam contratos determinísticos com encoders e runtimes injetados.
Não demonstram qualidade semântica de um modelo. Uma avaliação de qualidade precisa declarar
hipótese, baseline, variável alterada, seleção/anotações, métricas e critério de aceitação,
preservando os IDs dos artifacts dos três produtores e do scorer.
