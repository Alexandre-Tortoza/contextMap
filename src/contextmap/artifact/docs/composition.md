# Composição: geometria, entidades e relações

Este documento descreve `src/contextmap/artifact/composition.py`, `references.py` e `_invariants.py`. A proveniência (`origin`) e a linhagem estão em [`lineage.md`](lineage.md). O `ContextMap` conecta geometria autoritativa, entidades resolvidas e relações espaciais **sem duplicar os artifacts a montante e sem criar propriedade ambígua**.

```text
ContextMap
├── geometry_ref   -> GeometricMapArtifact (por identidade e tamanho)
├── entities[]     -> ContextEntity   ── geometry_refs[] -> GeometryReference
└── relations[]    -> ContextRelation ── subject/object   -> ContextEntityReference
```

## Decisões

| Pergunta | Decisão |
| --- | --- |
| A geometria é embutida, referenciada ou empacotada? | **Referenciada**. Nenhum ponto é copiado; um bundle é decisão de transporte do serializador e não muda o schema. |
| Escopo da identidade de entidade | `ContextEntityId` é único **dentro de um** `ContextMap`. A referência estável é `(context_map_id, entity_id)`. |
| Identidades de origem | Preservadas com os contratos reais: `ContextEntity.source` é um `ResolvedEntityReference` (run de resolução + id local) e `ContextRelation` guarda `source_run_id` (`SpatialRelationsRunId`) e `source_relation_id` (`RelationId`). O id no mapa **pode** diferir do id de origem, mas o mapeamento é sempre explícito; nada é reescrito em silêncio. |
| Sujeito e objeto de uma relação | `ContextEntityReference` para entidades **deste** mapa; a relação é direcionada (`sujeito predicado objeto`). |
| Índices | Derivados e reconstruíveis a partir das listas; nunca autoritativos (ver abaixo). |
| Estado não resolvido ou opcional | Preservado, nunca descartado nem promovido (ver abaixo). |

## Referências e escopos

- `ContextEntityReference` de outro mapa é recusada mesmo quando o id existe (`ForeignContextEntityReferenceError`), separando esse caso de "entidade inexistente" (`UnknownContextEntityError`). Equivalência entre mapas nunca é deduzida de ids com o mesmo texto;
- as referências a montante usam os tipos das capabilities donas: `ResolvedEntityReference` (Entity Resolution), `EntityReference` (Semantic Mapping, para as entidades de origem), `ResolutionDecisionId` (local ao run de resolução da entidade), `RelationId` e `SpatialRelationsRunId` (Spatial Relations). `UpstreamRecordRef(artifact_id, record_id)` continua existindo só para **citar evidência** heterogênea em uma `EvidenceOrigin` ([`lineage.md`](lineage.md)), onde não há um dono único de tipo;
- dois registros do mapa nunca mapeiam a mesma entidade resolvida nem a mesma relação a montante (a identidade seria ambígua): `ReferenceIntegrityError`.

## Entidade

`ContextEntity` guarda `entity_id`, `source`, `member_entities`, `resolution_decisions`, `unresolved_neighbors`, `geometry_refs`, `semantic_state` e `origin`. `member_entities` (pelo menos uma) e `resolution_decisions` preservam de quais entidades de origem e de quais decisões a entidade resolvida veio. `unresolved_neighbors` preserva a **incerteza de identidade**: entidades de origem fora desta que uma decisão deixou `UNRESOLVED` contra um membro (nem fundidas, nem declaradas distintas); nunca é também um membro. `origin` e a proveniência estão em [`lineage.md`](lineage.md).

- **Suporte de geometria autoritativo**: pelo menos um `GeometryReference`, ordenado e único, **nunca coordenadas**. O mapa valida que cada referência pertence ao mapa geométrico referenciado, usa a identidade canônica (`geometry_index_of`) e está dentro do intervalo `[0, point_count)`. Assim toda entidade resolve a suporte existente sem abrir a geometria; a verificação contra o `GeometricMap` real é do validador do artifact;
- várias entidades podem compartilhar elementos de geometria; o mapa não decide se isso é ou não um problema de resolução;
- resumos derivados (centroide, bounds) **não** fazem parte do schema: não têm consumidor hoje e seriam um segundo dado a manter consistente com a geometria. Podem ser adicionados depois como campo opcional (mudança aditiva, ver [`versioning.md`](versioning.md)).

### Estado semântico sem colapso de incerteza

`ContextSemanticState(status, hypotheses, support_status, primary_label, attributes)`; `LabelHypothesis` guarda o `label` verbatim (vocabulário aberto) e sua `origin` (como foi produzida e em que evidência se apoia). Não há score: os números que sustentam uma hipótese continuam no artifact a montante, que a origem cita.

| `AmbiguityStatus` | Hipóteses exigidas |
| --- | --- |
| `UNAMBIGUOUS` | exatamente uma |
| `AMBIGUOUS` | pelo menos duas, sem observações contraditórias |
| `CONFLICTING` | pelo menos duas, com observações que sustentam hipóteses incompatíveis |
| `INSUFFICIENT_EVIDENCE` | nenhuma; é abstenção, não evidência negativa |

O status precisa concordar com as hipóteses: um estado competitivo **não pode** ser reduzido a um rótulo pelo mapa. A ordem das hipóteses (por label) não é ranking.

### Crença pós-resolução e atributos declarados (schema `0.2.0`, opcionais)

Existência não é crença: uma entidade existe no mapa com todas as suas hipóteses, mas **qual** label ela é só aparece quando Entity Resolution derivou uma crença (`resolved-belief-min-support-v1`, contagem de observações físicas **distintas** por label sobre a união dos membros). Então:

| Campo | Significado |
| --- | --- |
| `support_status` (`ContextSupportStatus`) | `tentative` (nenhuma label atinge o mínimo), `supported` (exatamente uma atinge), `contested` (duas ou mais atingem) ou `insufficient_evidence` (sem hipótese). `None` quando não houve crença. Independe de `status`: uma entidade ambígua pode ter uma label sustentada, e as alternativas continuam listadas. |
| `primary_label` | A label sustentada, só quando `supported`; precisa ser uma das labels e ter a maior contagem. |
| `LabelHypothesis.supporting_physical_observation_count` | Quantas observações físicas distintas sustentam a label (grafias que só diferem tipograficamente compartilham a contagem), exatamente como a crença contou. Contagem, nunca score. Existe em toda hipótese quando há crença, e em nenhuma quando não há. |
| `attributes` (`ContextAttribute`) | Atributos declarados pelos claims (`category`, `color`, `material`…) ou derivados (`interaction_radius_m`): `name`, `value`, `attribute_origin` (`observed`/`derived`), `derivation_id` completo e `origin`. O `PolicyRef` de `origin` guarda a identidade e a versão da regra; o fingerprint depois de `@` em `derivation_id` identifica a configuração efetiva, sem fingir que é uma versão da regra. O raio cita as evidências fundidas do fator, as entidades semânticas de origem e as referências da geometria medida em `origin.derived_from`. Ordenados e únicos pela identidade completa `(name, value, attribute_origin, derivation_id ou "")`; valores iguais de origens ou derivações distintas coexistem. `class` fica fora porque `primary_label` da crença já declara o mesmo rótulo; conhecimento externo fica fora. |

Todos são omitidos do registro quando ausentes, então uma entidade sem crença e sem atributos é gravada exatamente como na `0.1.0` ([`versioning.md`](versioning.md)).

## Relação

`ContextRelation` guarda `relation_id`, `source_run_id`, `source_relation_id`, `subject`, `predicate`, `object`, `state`, `uncertainty_kinds` e `origin` (a evidência e a política que a derivaram).

- `origin` é **opcional** e só pode faltar numa relação `UNRESOLVED`: uma relação pode ter ficado indecidida justamente por não haver evidência nenhuma, e `None` diz isso em vez de citar o próprio registro da relação, o que a tornaria indistinguível de uma cuja evidência é exatamente esse registro. Uma relação decidida (`SUPPORTED` ou `REJECTED`) sem `origin` é recusada: decidir exige a evidência que decidiu. No registro, o campo ausente é a forma canônica de "sem proveniência" — nunca `null` ([`versioning.md`](versioning.md));

- sujeito e objeto resolvem a entidades **do mesmo mapa** (`ReferenceIntegrityError` caso contrário) e são distintos: uma relação de uma entidade consigo mesma não existe no schema;
- `state` é o `RelationState` **de Spatial Relations**: `SUPPORTED`, `REJECTED` (a evidência contradiz o predicado) ou `UNRESOLVED`. Uma relação candidata nunca é uma relação confirmada: relações não resolvidas ou rejeitadas **permanecem no mapa com seu estado**, e um consumidor que quer só as confirmadas filtra por `SUPPORTED`;
- `uncertainty_kinds` (`RelationUncertaintyKind`: `INSUFFICIENT_EVIDENCE`, `CONFLICTING_EVIDENCE`, `INCONSISTENT_STRUCTURE`) diz **por que** uma relação `UNRESOLVED` não foi decidida, como no contrato de origem: obrigatório e não vazio quando `UNRESOLVED`, vazio quando decidida. Assim evidência que se contradiz continua distinguível de evidência que falta;
- `predicate` é um `RelationPredicate` da taxonomia canônica de Spatial Relations (direção, simetria e versão da taxonomia pertencem a essa capability); um valor fora da taxonomia é rejeitado na decodificação. A política e a versão que derivaram a relação ficam em `origin.policy` (`decision_policy_id` e `taxonomy_version` do contrato de origem).

## Travessia sem consulta

`ContextMap.entity(reference)` e `ContextMap.relations_for(reference)` cobrem entidade → geometria (`geometry_refs`), relação → entidades (`subject`/`object`) e entidade → relações (como sujeito ou objeto, em ordem de `relation_id`). São mapas de busca em memória, derivados e reconstruíveis; **não há motor de consulta, grafo nem dependência de backend**, e um leitor leve implementa a mesma travessia sobre payloads preguiçosos.

## Capacidades versus conteúdo

`ContextMap` valida a declaração de `DeclaredCapabilities` contra o conteúdo real:

- entidades existentes exigem `ENTITIES` declarada; relações existentes exigem `RELATIONS`;
- com `RELATIONS` declarada, `relation_predicates` é **exatamente** o conjunto de `RelationPredicate` presentes, ordenado por valor;
- uma capacidade declarada com conteúdo vazio é válida (o estágio rodou e nada encontrou), e é diferente de uma capacidade ausente;
- cada capacidade também precisa do artifact que a produz na linhagem ([`lineage.md`](lineage.md)).

## Ordenação canônica

Entidades e relações são ordenadas por id e únicas; referências de geometria, por `(map_id, geometry_id)`. Dois mapas equivalentes codificam para o mesmo registro.
