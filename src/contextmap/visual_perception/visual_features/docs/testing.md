# Testing: visual_features

Os testes desta capability **ainda vivem em `tests/visual_perception/`**, sob o owner anterior. A
decomposição moveu o código e não moveu os testes; não existe `tests/visual_features/`.

## Responsabilidade → teste

| Responsabilidade | Arquivo de teste | Testes |
|---|---:|---:|
| Identidade do espaço de embedding e compatibilidade | `tests/visual_perception/test_embedding_space.py` | 8 |
| Persistência, índice e integridade do payload | `tests/visual_perception/test_feature_store.py` | 16 |
| Associação região ↔ grade densa e pooling | `tests/visual_perception/test_dense_region_association.py` | 18 |
| Fronteira validada do realce (dez invariantes) | `tests/visual_perception/test_feature_resolution_enhancement.py` | 18 |
| Diagnósticos de extração | `tests/visual_perception/test_feature_diagnostics.py` | 80 |

### Adapters

| Adapter | Teste de contrato | Teste de runtime |
|---|---:|---|
| `siglip2` | `backends/test_siglip2.py` (47) | `backends/test_huggingface_runtimes.py` |
| `alphaclip` | `backends/test_alphaclip.py` (35) | idem |
| `clip` | `backends/test_clip.py` (28) | idem |
| `dinov2` | `backends/test_dinov2.py` (23) | idem |
| `dinov3` | `backends/test_dinov3.py` (20) | `backends/test_dinov3_validation.py` |

O par é deliberado: o teste de contrato exercita o adapter contra o port com fakes e sem baixar
modelo; o de runtime exercita a ponte com o SDK e é saltado quando a dependência não está
instalada.

## Níveis

| Nível | Onde | O que protege |
|---|---|---|
| unit | `test_embedding_space`, `test_dense_region_association` | cada contrato e o pooling isoladamente |
| contract | `backends/test_{dinov2,dinov3,siglip2,clip,alphaclip}.py` | os cinco extratores respeitam o mesmo port e declaram escopo coerente |
| contract | `test_feature_resolution_enhancement` | os dez invariantes do realce, com enhancer fake |
| integration | `test_feature_store` | writer, índice e reader fecham o ciclo com integridade |
| integration | `tests/visual_perception/test_run_artifact.py` | o feature store dentro do artifact e seu inventário |
| architecture | `tests/architecture/test_boundaries.py` | direção de dependências, API pública, import privado |
| architecture | `tests/architecture/test_compatibility_facade.py` | todo símbolo da fachada resolve e é o mesmo objeto do owner |
| end-to-end | `tests/end_to_end/test_perception_pilot_harness.py` | o pipeline inteiro, com os extratores compostos |

## Invariantes que nunca podem quebrar

1. **Mesma dimensão não é mesmo espaço.** O fingerprint cobre os sete campos de `EmbeddingSpace`.
   Dois modelos de 768 dimensões têm fingerprints distintos, e comparar seus vetores é
   `EmbeddingSpaceMismatchError`. Protegido por `test_embedding_space`.
2. **O array nunca entra no contrato público.** `VisualFeature` carrega `payload_reference`. Se um
   campo com o vetor aparecer, o artifact deixa de ser legível sem carregar tudo em memória.
3. **A região é congelada.** O pooling mapeia geometria e agrega; não altera a `Region2D` recebida.
4. **Região sem suporte é erro, não zero.** `EmptyRegionSupportError` quando nenhuma célula da
   grade cai na região. Um vetor de zeros seria indistinguível de uma feature legítima.
5. **A política de pooling está na proveniência.** `mask_weighted_mean_preserve_l2_v2` é nomeada e
   versionada; duas políticas diferentes produzem proveniências diferentes.
6. **O realce produz identidade nova.** `feature_id`, `payload_reference` e — sob normalização
   diferente — `embedding_space_id` distintos da origem, com lineage nomeando o mapa exato. São os
   dez invariantes de `feature_enhancement.py`.
7. **O payload preserva o array exatamente.** Shape, dtype e byte order não nativo sobrevivem ao
   round-trip, porque o contêiner é `.npy` nativo.
8. **A referência não escapa da raiz do store.** `_resolve_within_root` recusa caminho que saia do
   diretório, e chave ou `payload_reference` duplicados são erro.
9. **O índice tem schema versionado.** `FEATURE_INDEX_SCHEMA_VERSION = "0.1.0"`; mudar o formato
   exige mudar a versão.

## Regressões protegidas

| Falha observada | Teste |
|---|---|
| `feature_enhancement` reduzido a stub, perdendo as dez validações | `test_feature_resolution_enhancement.py` (18 testes; não era coletável, por isso a perda passou) |
| `EMBEDDING_SPACE_SCHEMA_VERSION` perdido na migração | `tests/packaging/test_release_scope.py` |
| `feature_store` reduzido a stub sem validação nem containment | `test_feature_store.py` (16 testes) |
| Espaços de embedding de mesma dimensão tratados como compatíveis | `test_embedding_space.py` |

As três primeiras são do mesmo padrão: um módulo foi deletado e substituído por versão mais pobre,
e o teste que o protegia não coletava. Vale como aviso de método — um teste que não coleta não
protege nada.

## Lacunas conhecidas

- **Não há `artifacts/` para testar.** O writer e o reader de run que existiam aqui foram
  removidos: não tinham consumidor em `src/` nem em `tests/`, e este pacote é componente do
  stage `visual_perception`, não um stage. As features são publicadas no artifact daquele
  stage, cujos testes estão em `tests/visual_perception/`.
- **Não há backend real de realce.** Os 18 testes de `feature_resolution_enhancement` usam
  enhancer fake. Protegem a fronteira, não uma implementação.
- **As torres de texto saíram deste pacote.** Foram para `semantic_scoring.adapters`, e os
  testes delas (`backends/test_clip_text.py`, `backends/test_alphaclip_text.py`) estão mapeados
  em `semantic_scoring/docs/testing.md`.
- **Os testes não foram movidos.** Enquanto viverem em `tests/visual_perception/`, a
  correspondência entre capability e suíte depende deste documento em vez da estrutura de
  diretórios.
