# Execução dos produtores de percepção

O DAG do runtime possui três stages independentes:

```text
region_perception: discovery + grounding opcional + refinamento opcional → regiões finais
visual_perception: features densas/globais/regionais + pooling opcional
semantic_interpretation: interpretação de cena e região conforme a política de request
```

Visual Perception e Semantic Interpretation consomem o artifact regional selecionado e a
mesma sequência. Seus manifests identificam explicitamente os upstreams, seleção e código.
Scoring seleciona os três produtores separadamente. Nenhum executor replica o artifact vizinho.

## Grafos locais

`execution/pipeline_resolution.py` possui `StageSpec`, `PipelinePreset`, validação, resolução
de backends e dispatch dos ports. `execution/pipeline_presets.py` possui presets e suas
derivações; presets locais não substituem o DAG de artifacts do runtime. O runtime compõe
os grafos apropriados à responsabilidade de cada produtor.

`ResolvedPipeline.build_stage_graph()` recebe fontes explícitas, inclusive imagem preparada
e regiões finais quando necessárias. O grafo registra `StageOutcome` com status, duração e
erro. Falhas deixam branches independentes executáveis; dependentes são ignorados explicitamente.
O artifact mantém cobertura de observações mesmo quando o envelope produzido fica vazio.

## Agendamento e ciclo de vida

`frame_major` executa o grafo de cada observação. `stage_major` agrupa a execução pelo estágio,
liberando cada backend depois de seu último consumidor. A preparação é consistente com o
frame regional, e máscaras são lidas pelo resolver do upstream.

Arrays de features são gravados imediatamente em staging. O cache numérico retém somente a
observação corrente; pooling de observações anteriores carrega um payload verificado por hash,
sem nova inferência. `finalize()` valida payloads e diagnósticos, inventaria o conteúdo e publica
atomicamente. Falhas de publicação descartam o staging sem alterar artifacts existentes.

Veja [composição global](../../../../docs/runtime-composition.md),
[artifact visual](visual-evidence-artifact.md) e [invariantes](scientific-invariants.md).
