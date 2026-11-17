# Fluxo de dados — Region Perception

```text
SequenceArtifact + seleção explícita
    → PreparedImage
    → descoberta por passes
    → normalização e audit
    → namespace_regions(run regional, observação)
    → grounding opcional: requests → outputs caixas/pontos
    → refinamento opcional: propostas → prompts → máscaras/outcomes
    → assemble_regions(region-set-assembly/1)
    → externalização das máscaras
    → RegionPerceptionRunArtifact
```

A normalização mantém identidades locais e contratos usados nos goldens. A fronteira de
publicação qualifica cada região descoberta com run regional, observação e índice, remapeando
os links dos hints nativos na auditoria. Candidatos e decisões de merge permanecem chaves
locais de audit, sem alteração científica na geometria ou nas regras de normalização.

Grounding e refinamento carregam `region_perception_run_id` no request. Seus IDs de request
continuam sendo digests dos inputs de inferência: a mesma consulta sobre o mesmo conteúdo
mantém esse digest em outro run, enquanto os IDs da evidência produzida mudam com o produtor.
Caixas, pontos e máscaras refinadas têm namespaces distintos. Refinamento mantém a identidade
da proposta como contributor e não substitui sua evidência.

`assemble_regions` concatena descobertas, caixas de grounding por query e regiões refinadas
por prompt. Não deduplica nem funde regiões. Uma colisão de identidade é erro explícito.

`write_frame` registra inclusive ausência de regiões e falha. `write_grounding` conserva
request, prompt, raw response, outputs e diagnósticos. `write_refinement` externaliza máscaras
e devolve a execução com referências para que os outcomes e o conjunto canônico usem o mesmo
payload. `finalize` exige cobertura exata da seleção e publica tudo atomicamente.

A leitura fornece `read_frames`, `read_regions`, `read_region_discovery_audits`,
`read_grounding`, `read_refinement` e `mask_resolver`. Falha de acesso a uma máscara não vira
uma região box-only silenciosamente.

A publicação exige que cada caixa aceita de grounding e máscara refinada apareça no conjunto
canônico. Cada prompt refinado deve coincidir com o output de grounding que nomeia, inclusive
geometria e provenance. Referências de máscara pertencem ao store do produtor; pixels staged
sem uma região publicada ou constraint de preparo impedem a finalização.

`iter_frames()` hidrata constraints de um frame por vez. `frame(source_observation_id)`
carrega apenas o frame selecionado, sem materializar os rasters das outras observações.
`resolve_mask(reference)` no writer permite auditar uma máscara staged antes da publicação;
reapresentar os mesmos pixels reutiliza sua referência, e pixels conflitantes falham.
