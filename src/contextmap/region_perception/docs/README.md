# Region Perception

A capability produz candidatos espaciais 2D, normaliza a descoberta e executa grounding e
refinamento opcionais. Uma região preserva geometria, observação física e provenance; não
implica uma classificação semântica ou uma entidade persistente.

O stage `region_perception` publica o conjunto final e auditável de regiões como
`RegionPerceptionRunArtifact`. Extração de features e interpretação semântica selecionam esse
artifact independentemente. Trocar esses consumidores preserva as identidades regionais e não
reexecuta descoberta, grounding ou refinamento.

## Contratos e leitura

A API pública está na raiz `contextmap.region_perception`:

- `Region2D`, `RegionId`, geometria e provenance regional;
- ports de descoberta, grounding e refinamento;
- `namespace_regions` para qualificar regiões normalizadas pelo run e observação;
- `assemble_regions` com a política `region-set-assembly/1`;
- `RegionPerceptionRunManifest`, `RegionPerceptionRunWriter` e `RegionPerceptionRunReader`.

O schema publicado é `0.2.0`. A seleção explícita inclui `sequence_artifact_id`,
`sequence_name`, `selection_id` e `source_observation_ids`. Toda observação selecionada possui
um `RegionPerceptionFrame`: preparo de imagem, outcome `succeeded` ou `failed` e detalhe da
falha quando aplicável. Um frame bem sucedido sem regiões continua distinguível de uma falha.

Writer e reader preservam auditoria de descoberta, requests e execuções de grounding,
respostas textuais exatas, prompts e outcomes de refinamento, além das máscaras. A publicação
é atômica, inventariada por hashes e imutável. Máscaras de regiões e de constraints de preparo
ficam em payloads separados; JSON contém referências, não pixels.

## Documentação

- [Arquitetura](architecture.md): ownership, dependências e invariantes.
- [Fluxo de dados](data-flow.md): da observação ao artifact.
- [Mapa de código](code-map.md): responsabilidades dos módulos.
- [Testes](testing.md): garantias e validação.

- [Descoberta](region-discovery.md): passes, normalização e auditoria.
- [Grounding](region-grounding.md): consultas, identidade e resposta nativa.
- [Refinamento](region-refinement.md): aceitação de máscaras e linhagem de propostas.
- [Máscaras](mask_store.md): payloads compactos, staging e leitura lazy.
