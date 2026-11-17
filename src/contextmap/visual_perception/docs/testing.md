# Testes visuais

As suites de Visual Perception preservam contratos, identidade, espaços de embedding,
geometria densa, pooling, payloads, diagnósticos e lifecycle sem carregar modelos reais.
Discovery, grounding e refinamento são verificados nas suites do produtor regional;
requests, prompts, views e parsing nas suites do produtor semântico.

Os testes de artifact visual cobrem round-trip, upstreams explícitos, ausência de cópias de
regiões/máscaras/claims, integridade lazy, publicação sem substituição, cleanup de staging e
payloads ligados exatamente às features. Diagnósticos densos são cruzados com a feature
persistida; debug não é usado para reconstruir geometria.

Na integração de runtime, os testes verificam os três produtores sobre as mesmas regiões,
falhas isoladas, schedules, release de backends, vida limitada dos arrays, scoring independente
e canais densos opcionais de Sensor Association. A cadeia sintética verifica associação,
fusão, entidades e artifact final; ela é evidência de contrato, não medição de qualidade real.

```bash
pytest tests/visual_perception tests/unit/visual_features
pytest tests/runtime/test_independent_perception_executors.py
pytest tests/runtime/test_runtime_dense_channels.py tests/runtime/test_runtime_stage_major_dense_lifetime.py
pytest tests/end_to_end/test_runtime_chain.py
make check
```

Mudanças determinísticas seguem Red → Green → Refactor. Avaliações com modelos reais precisam
de hipótese, baseline, seleção, métrica e critério antes da execução. Evidência histórica
registrada permanece datada e não demonstra automaticamente o novo layout dos artifacts.
