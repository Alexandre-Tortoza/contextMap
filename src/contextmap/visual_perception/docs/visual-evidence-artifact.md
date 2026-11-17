# Artifact de evidência visual

`VisualEvidenceArtifact` é o produto persistido de `visual_perception`. A schema
`0.1.0` separa features da descoberta regional e da interpretação semântica. O
manifest declara `region_artifact_id`, `sequence_artifact_id`, `selection_id`, a
identidade do produtor, configuração efetiva e identidade do código. O artifact
não copia regiões, máscaras nem claims dos produtores selecionados.

`VisualEvidenceResult` agrupa features por `source_observation_id`; não possui
identidade própria de resultado. Um envelope vazio registra que aquela observação
foi processada. Os outcomes de estágio conservam status, duração e erros; o
executor fornece a identidade da observação ao registrar outcomes por frame.
`IneligibleRegion` registra explicitamente cada skip regional por estágio.

## Publicação e leitura

`VisualEvidenceRunWriter` recebe identidade e upstreams explícitos. Os métodos
`add_feature_payload`, `add_feature_diagnostic` e `add_feature_preview` preservam
os canais da extração. Arrays são gravados imediatamente em staging e não ficam
retidos pelo writer. `add_result` retém somente metadados leves. Ao finalizar, o
writer valida que cada payload/diagnóstico produzido pertence exatamente a uma
feature, verifica geometria densa e fontes do pooling, inventaria hashes e publica
atomicamente sem substituir um diretório existente. Sair do contexto sem
`finalize()` remove o staging.

`VisualEvidenceRunReader` abre os metadados sem carregar arrays. `iter_results()`
decodifica uma linha por vez. `feature_store()` abre o índice sob demanda; os
payloads são carregados individualmente e verificados por hash. `dense_feature_map`
reconstrói o sampling através dos diagnósticos contratuais, sem depender de debug.

## Layout contratual

- `manifest.json`: schema, contrato, lineage, seleção e inventário;
- `outputs/results.jsonl`: envelopes tipados por observação;
- `outputs/features/`: índice e payloads `.npy`;
- `outputs/ineligible-regions.jsonl`: skips regionais explícitos, somente quando existem;
- `metrics/stage-timings.jsonl`: outcomes leves de execução;
- `metrics/feature-extraction.jsonl`: métricas e geometria do sampling quando produzidos.

`__DEBUG__/` contém somente diagnósticos humanos, controlados por
`FeatureDebugLevel`. Os metadados de imagem nos envelopes usam referências à
observação física; caminhos temporários são removidos. Máscaras de elegibilidade
continuam no artifact regional upstream, e não são replicadas como evidência visual.

No agendamento `stage_major`, o runtime conserva no cache somente os arrays da observação corrente. Pooling de uma observação anterior relê um único payload em staging com verificação de hash; os arrays são liberados depois do consumidor. A identidade de cada feature combina o run visual, a observação física, o produtor de features e o índice de emissão.
