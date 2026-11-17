# Protocolo de interpretação semântica

Semantic Interpretation recebe uma imagem e as regiões finais de um produtor regional
explicitamente selecionado. Produz hipóteses, contexto, abstenções e failures auditáveis;
a resposta de um modelo nunca se torna crença persistente por si só.

## Evidência e identidade

Requests e claims registram `semantic_interpretation_run_id` e `source_observation_id`.
Claims regionais referenciam regiões finais do artifact regional; contexto de cena possui
`context_id`. O manifest conserva sequência, seleção e `region_artifact_id`.
Features não são upstream implícito dessa execução. Variar o artifact visual não altera
a identidade de um produtor semântico já selecionado.

As views escolhidas pela política são materializadas e content-addressed. O SHA-256
identifica os bytes exatos enviados ao backend; runtimes verificam esses bytes antes de
abrir imagens ou chamar providers. Máscaras e elegibilidade vêm do produtor regional.
Uma caixa não satisfaz silenciosamente uma política que exige máscara; skips preservam
seu motivo como evidência contratual.

## Prompt, parsing e contexto

O prompt é identificado por template, schema e fingerprint do texto renderizado.
O parser mantém hipóteses primárias e alternativas, atributos, abstenção e respostas
rejeitadas explicitamente. `UNSCORED_ONLY` não converte declarações de confiança do VLM
em probabilidades; `MEASURED` só é usado quando o backend realmente fornece o score sob
semântica declarada. Ausência de score não significa zero.

Condicionamento por cena é uma política explícita: um request regional recebe exatamente
o contexto da mesma observação e do mesmo produtor semântico, com sua referência e
renderização versionada. O grafo possui essa dependência quando configurada. Orçamento
`largest-area-regions-v1` limita requests pelas áreas de suporte, com desempate estável;
regiões excluídas do orçamento continuam no artifact regional.

## Um registro auditável por desfecho

`SemanticInterpretationService` constrói requests/views e aplica a política. O writer
registra observações, execuções materializadas, respostas rejeitadas pelo parser e
falhas de chamada em streams separados, junto com as views exatas. Suas validações
cruzam claims, contexto, requests, hashes e provenance. Falha não vira abstention nem
inferência silenciosa de outro backend. Debug é diagnóstico humano, não fonte contratual.

Comparações científicas devem usar as mesmas observações, regiões finais, modos e
variantes de evidência. Qualidade e custo permanecem métricas diferentes. Scoring de
hipóteses é outra capability e pode reutilizar esse artifact sem rodar o VLM novamente.

O [registro pré-decomposição](history/pre-decomposition-interpretation.md) conserva os
protocolos detalhados de views, templates/parser, Qwen, Gemini e Florence-2, suas
adaptações upstream e a evidência científica original. Esse registro identifica
explicitamente os contratos antigos; o código e a [API ativa](README.md) definem o
comportamento presente. [Eagle 2.5](eagle2_5.md) possui uma nota ativa própria.
