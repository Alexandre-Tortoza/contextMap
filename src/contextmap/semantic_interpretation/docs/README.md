# Semantic Interpretation

Esta capability interpreta imagens e regiões congeladas como **hipóteses semânticas**, sem
transformar a resposta de um modelo em crença persistente. O stage publica um
`SemanticInterpretationArtifact` selecionável independentemente do artifact visual.

`SemanticClaim` e `SceneContext` pertencem a esta capability. Cada evidência registra
`semantic_interpretation_run_id`, `source_observation_id`, a região quando aplicável, e a
proveniência do modelo, prompt, política e resposta bruta. O contexto também possui `context_id`.
IDs de claims e contexto incluem produtor, observação e request; variar features não os altera.

O stage consome a sequência e o `RegionPerceptionRunArtifact` explicitamente selecionados.
`SemanticInterpretationService` recebe o interpretador, a política, o diretório de imagens
preparadas e o resolver das máscaras regionais. Ele constrói views e requests, preserva falhas
e condiciona requests regionais ao contexto da mesma observação quando configurado. Runtime
seleciona e constrói backends; não decide essas regras científicas.

O artifact preserva, em streams contratuais separados:

- observações, inclusive saídas vazias, falhas e skips explícitos;
- execuções bem-sucedidas, com prompt, resposta bruta, parse e diagnósticos;
- respostas observadas rejeitadas pelo parser;
- chamadas que falharam antes de responder;
- bytes exatos de cada view e razões de inelegibilidade de regiões.

O writer descarrega views para staging imediatamente, verifica referências e hashes e publica
atomicamente. O reader verifica inventário e streams; ausência não se torna saída vazia.
`__DEBUG__/` contém apenas diagnósticos humanos e nunca é dependência downstream.

A API pública é `contextmap.semantic_interpretation`; adapters concretos ficam internos.
Não há reexports do antigo subpacote de Visual Perception.

- [Arquitetura e fronteiras](architecture.md)
- [Fluxo, identidade e persistência](data-flow.md)
- [Mapa do código](code-map.md)
- [Testes e avaliação](testing.md)
