# Arquitetura — Region Perception

## Ownership

A capability possui os conceitos que **introduz**: a identidade de uma região, sua
geometria, a decisão de aceitá-la ou rejeitá-la, e a persistência das máscaras.

| Conceito | Onde vive | Observação |
|---|---|---|
| `RegionId`, `Region2D` | `region_evidence.py` | contrato canônico de uma região aceita |
| Descoberta e normalização | `discovery/` | passes, merge, rejeição, audit |
| Grounding | `grounding/region_grounding.py` | query de texto como entrada persistida |
| Refinamento | `refinement/region_refinement.py` | máscara a partir de prompt geométrico |
| Elegibilidade | `region_eligibility.py` | por que uma região não serve a um estágio |
| Hints semânticos | `region_semantic_hint.py` | texto nativo do backend, como evidência |
| Persistência de máscara | `artifacts/mask_store.py` | payload `.npy`, índice, integridade |
| Ports | `ports.py` | quatro pontos de substituição reais |

A capability **não** possui: features visuais por região (`visual_features`),
interpretação semântica (`semantic_interpretation`), score de compatibilidade
(`semantic_scoring`) e identidade persistente de entidade (`semantic_mapping`,
`entity_resolution`).

## Direção de dependências

As dependências permitidas são declaradas em `tests/architecture/test_boundaries.py`
(`ALLOWED_DEPENDENCIES`) e verificadas por teste:

```text
region_perception → shared, ingestion, model_runtime, visual_features, visual_perception
```

O sentido acima é *consumidor → API pública do produtor*, o inverso do fluxo de dados.

**Proibido**, e verificado: importar `contextmap.runtime` (regra `runtime-isolation`),
importar caminho interno de outra capability em vez da fachada
(`public-api-boundary`), e importar módulo privado ou pacote `adapters` de outra
capability (`private-cross-module-import`).

Duas dependências acima são **dívida declarada**, não desenho:

- `visual_perception` — o contrato `PreparedImage` continua sediado em
  `visual_perception.models`, e é correto por ownership: é a saída da preparação de
  imagem, não um conceito de região. Os codecs e geradores de identidade de região agora pertencem a esta capability,
  sem import inverso privado.
- `visual_features` — dependência mútua com ela. Os dois sentidos estão corretos por
  ownership (`Region2D` é daqui, `FeatureId` é de lá); é dependência de tipo entre
  capabilities, não inversão de camada.

`BackendProvenance` já migrou para `contextmap.shared`, e com ele `ArtifactReference` e
`JsonScalar`, que não eram contrato de região nem de percepção. `BoundingBox`,
`BoundingBox2D`, `InlineMask`, `CoordinateConvention`, `RejectionReason`, `BackendScore`,
`RegionProvenance`, `NativeRegionText`, `RegionCandidate` e `RejectedRegionCandidate`
vieram para `region_geometry.py`, nesta capability, que é a dona do conceito de região;
`visual_perception.region_models` só reexporta os nomes antigos, apontando para estes
mesmos objetos.

## API pública

A fachada é `src/contextmap/region_perception/__init__.py`, com 69 símbolos em
`__all__` — grande para o que §17 do AGENTS.md pede, e expõe internos de descoberta e
normalização (`PassKind`, `DiscoveryPassConfig`, `MergeKind`, `NormalizationConfig`).
Encolher isso depende de saber quais consumidores dependem de quê; a dívida está
registrada, não resolvida.

Um ponto a conhecer: `RegionCandidate` é reexportado de `region_geometry`, **não** de
`region_evidence`. Existem duas definições com esse nome — ver *Invariantes*.

## Ports e adapters

Cada `Protocol` em `ports.py` corresponde a uma variação real:

| Port | Implementações hoje |
|---|---|
| `RegionDiscovery` / `AuditedRegionDiscovery` | `adapters/sam3.py`, `adapters/florence2.py` |
| `RegionGrounding` | `adapters/locateanything.py` |
| `RegionRefinement` | `adapters/sam2.py` |

`RegionRefinement` e descoberta SAM2 são implementados em `region_perception/adapters/sam2.py`.
A configuração específica fica interna; runtime compõe os backends.

Nenhum port constrói backend nem importa SDK de modelo. O carregamento HuggingFace e a
verificação de device vêm de `contextmap.model_runtime`, infra declarada como tal.

## Composição no runtime

O stage regional executa descoberta, grounding e refinamento opcionais e publica o conjunto
final de regiões. Extração de features e interpretação semântica consomem esse artifact como
ramos independentes. Isso substitui a decisão temporária do passo 1 do ADR, em que somente a
descoberta rodava neste stage.

O artifact `0.2.0` contém seleção explícita, frames com metadados de preparo e outcomes,
auditoria da descoberta, grounding com raw responses exatas, refinamento com scores nativos,
regiões canônicas e máscaras hash-verificadas. Runtime coordena execução e lifecycle; as
regras científicas de normalização, aceitação e `region-set-assembly/1` pertencem aqui.

`namespace_regions` preserva os contratos locais do normalizer e qualifica identidades na
publicação pelo produtor regional e pela observação. O wire format da auditoria usa
`region_artifact_id` e `source_observation_id`, sem identidade de resultado fundido. Máscaras
são externalizadas depois desse remapeamento. Grounding e refinamento qualificam identidades
por request ou proposta, além do produtor e observação.

## Invariantes arquiteturais

**Ausência nunca é fabricada.** `source_observation_id` é `None` quando desconhecido, e
isso é um estado válido. Inventar um id de observação transformaria "não sei de onde veio"
em "veio de outra observação", e os consumidores que confrontam região com frame rejeitam
id divergente mas aceitam `None`.

**Score não é campo da região.** Um score de backend tem semântica própria — detector,
similaridade, qualidade geométrica não são a mesma grandeza — e fica onde foi medido:
auditoria de descoberta, diagnósticos de grounding e refinamento. Nunca promovido a número
na `Region2D` que um consumidor possa ler como probabilidade (§5).

**Região é evidência, não crença.** Uma `Region2D` afirma extensão espacial e
proveniência. Não afirma identidade semântica — isso vem de claims e features. Um hint
de texto do backend (`region_semantic_hint.py`) é evidência de correlação visual, não
rótulo verdadeiro.

**Rejeição é explícita.** `is_accepted=False` exige `rejection_reason`; o inverso
também é validado. As rejeições e decisões de merge chegam ao artifact em vez de
morrerem com a execução.

**Falha de acesso ≠ ausência.** `region_mask_pixels` levanta `MaskStoreError` quando a
região tem `mask_reference` e nenhum resolver foi dado. Devolver `None` ali apagaria a
diferença entre "não tem máscara" e "tem máscara persistida que não consegui ler", e o
chamador seguiria tratando a região como box-only.

**O payload de máscara é um `.npy` real.** O envelope `np.save` carrega dtype e shape,
o hash SHA-256 cobre o arquivo como ele chega ao disco, e
`MASK_INDEX_SCHEMA_VERSION = "0.1.0"` é o schema congelado pela release v0.1.0.

### Dívidas que o código carrega

| Dívida | Onde | Efeito |
|---|---|---|
| Duas definições de `RegionCandidate` | `region_evidence.py` (reduzida) e `region_geometry.py` (rica, 14 campos) | a pipeline de normalização consome a rica; a reduzida não tem consumidor de produção |
| `BackendScore` reduzido a `float` | `discovery/region_normalization.py` | `name` e `semantics` do score nativo são descartados; o que chega à auditoria já perdeu a semântica do backend |


As duas primeiras foram apontadas em review. Três dívidas que estavam aqui saíram: o stage
declarado sem executor (há `RegionPerceptionExecutor`, e um plano que o alcance roda),
`to_dict()` não mais exige origem conhecida (a asserção foi removida, e a ausência viaja
como `null`), e o artifact passou a ter publicação atômica, `schema_version`, inventário
com SHA-256 por arquivo contratual e fronteira tipada (`Region2D` na entrada e na saída).

## Documentos relacionados

- `README.md` — visão geral e API pública
- `code-map.md` — arquivo a arquivo, com o que cada um não deve conter
- `data-flow.md` — os dados reais através da implementação
- `testing.md` — responsabilidade → teste, e os invariantes protegidos
