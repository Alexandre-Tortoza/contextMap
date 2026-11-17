# Fluxo e persistência

```text
SequenceArtifact + RegionPerceptionRunArtifact explicitamente selecionados
    -> PreparedImage + regiões finais + resolver das máscaras upstream
    -> SemanticInterpretationService
    -> views exatas + SemanticInterpretationRequest
    -> SemanticInterpreter
    -> execução / rejeição de parse / falha de chamada
    -> SemanticObservation
    -> SemanticInterpretationArtifact
```

O contexto de cena é produzido primeiro, quando habilitado. Os requests regionais seguem a
política de views, ordenação e orçamento. Uma região sem máscara sob uma política que precisa
da máscara recebe um registro de inelegibilidade; nenhuma máscara é fabricada a partir da caixa.
Cada request regional tem desfecho independente: uma rejeição preservada não descarta as
respostas das outras regiões. Uma falha de cena permanece falha do seu estágio.

## Identidade e proveniência

Claims e contexto preservam o produtor semântico e a observação física. O parser qualifica IDs
por run, observação, request e índice. `SceneContext.context_id` é a chave usada pelo
`scene_context_reference`, nunca o ID de um resultado fundido. Proveniência distingue backend,
política, prompt, schema e resposta bruta; confiança ausente continua `None`.

No `finalize`, claims e contexto apontam para `outputs/semantic-interpretations.jsonl`, que
contém a resposta textual completa. Falhas de parse apontam para seu próprio stream. Uma
chamada sem resposta não reivindica `raw_response_reference`.

## Layout contratual

| Caminho | Conteúdo |
| --- | --- |
| `outputs/semantic-observations.jsonl` | Cobertura e evidências por observação |
| `outputs/semantic-interpretations.jsonl` | Execuções bem-sucedidas e respostas brutas |
| `outputs/semantic-interpretation-failures.jsonl` | Respostas rejeitadas, com hash e motivo |
| `outputs/semantic-backend-call-failures.jsonl` | Requests sem resposta e erro do fornecedor |
| `outputs/semantic-views/` | Bytes exatos das views com hash no inventário |
| `outputs/ineligible-regions.jsonl` | Regiões não solicitadas e motivos explícitos |
| `metrics/stage-timings.jsonl` | Status, duração e falhas de estágios |

Streams de falhas existem mesmo vazios. Os payloads são gravados diretamente em staging,
referências/hash são verificados antes da publicação e o diretório final aparece por rename
atômico. Configurações persistidas e diagnósticos passam por redação de credenciais. O writer
recusa sobrescrever um artifact finalizado.
