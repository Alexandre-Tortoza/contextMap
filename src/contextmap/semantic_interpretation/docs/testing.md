# Testes e avaliação

A suíte desta capability vive em `tests/semantic_interpretation/`, com backends em
`backends/`. Os testes migrados preservam as invariantes dos contratos e dos adapters existentes.

## Cobertura determinística

- Contratos: hipóteses primárias/alternativas, score ausente, atributos, observação e produtor.
- Requests: modos, scopes, views, embedding spaces e contexto resolvido da mesma observação.
- Views: hashes, pixels exatos, contorno exterior, crops, preenchimento e política versionada.
- Prompts/parser: instruções e schemas publicados, abstenção explícita, confiança e falhas.
- Backends: runtime/client injetado, capabilities, dispositivos, parâmetros e release.
- Artifacts: round-trip tipado, upstream regional, inventário, publicação atômica, imutabilidade,
  payload ausente/corrompido, materialização dos outputs e resposta bruta contratual.
- Falhas: parse rejeitado e chamada sem resposta permanecem streams distintos e completos;
  streams vazios nunca se confundem com ausência de tracking.

Os fingerprints históricos de prompts continuam como guardas de instruções/schema. Os testes
normalizam **somente** o header de identidade de request (rename e ordenação JSON), preservando
o digest original; também verificam que o texto atual usa o produtor semântico e que seu hash
mede exatamente os bytes atuais. Nenhum golden científico foi regravado.

```bash
pytest -q tests/semantic_interpretation -p no:cacheprovider \
  --basetemp="$HOME/.cache/cm2/semantic-tests"
```

## Limite científico

As mudanças de decomposição são determinísticas; não mudam modelo, checkpoint, instrução,
política de seleção ou regra de interpretação. A validação usa fixtures e runtimes injetados,
sem chamadas pagas nem downloads. Qualidade semântica de modelos exige avaliação própria com
hipótese, baseline, seleção/anotações e métrica; os testes de contrato não demonstram acurácia.
