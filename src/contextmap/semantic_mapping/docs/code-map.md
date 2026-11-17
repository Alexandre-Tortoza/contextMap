# Mapa de código de Semantic Mapping

Este documento registra o ownership dos arquivos da capability. Ele é o contrato arquitetural para revisões futuras: se uma função nova não couber na responsabilidade declarada do arquivo, isso é uma discussão de ownership, não um lugar para encaixá-la. Tamanho de arquivo é um sinal, não o critério de divisão, e números de linha não fazem parte do contrato.

A capability tem cinco camadas: **vocabulário** (primitivas sem semântica de contrato), **contratos** (o que uma entidade afirma), **políticas** (as regras versionadas que as produzem), **auditoria** (conferir e percorrer evidência já persistida) e **persistência** (como um run é gravado e reaberto). Nenhuma camada importa a de cima, e a persistência **nunca** importa política.

`tests/architecture/test_semantic_mapping_layering.py` torna este documento executável: ele recusa um import para cima, exige que todo módulo e subpackage novo declare camada, e proíbe a fachada de dentro da capability.

## Vocabulário

| Arquivo | Responsabilidade | Entrada → saída | Conceitos | Não deve possuir |
| --- | --- | --- | --- | --- |
| `invariants.py` | As invariantes que mais de um contrato precisa afirmar. | contrato em construção → erro ou silêncio | `require_present`, `require_count_of_at_least_one`, `require_canonical` | Construir contrato, decidir política, escolher limiar, ter side effect, aparecer na fachada. |
| `fingerprint.py` | Os digests de identidade da capability e a sua forma canônica. | registro ou conjunto de ids → `sha256:…` | `canonical_digest`, `geometry_set_digest` | Montar o registro a digerir, conhecer contrato, decidir o que entra num fingerprint. |
| `support_shape.py` | A **forma** de um suporte de pontos: o mecanismo, que muda por custo. | coordenadas + números → coordenadas e números | centroide (`fsum`), bounds, componentes conexos (grade de 27 células + union-find), eixos principais (Jacobi), `DEFAULT_MAX_CONNECTIVITY_POINTS`, `PrincipalAxes` | Conhecer contrato, política ou provenance desta capability; escolher limiar científico; construir `EntityOrientation`. |
| `fused_evidence_source.py` | A porta de leitura de um run de fusão persistido. | — → `Protocol` | `FusedEvidenceSource` | Implementação concreta, filesystem, política de validação. |

## Contratos

| Arquivo | Responsabilidade | Entrada → saída | Conceitos | Não deve possuir |
| --- | --- | --- | --- | --- |
| `entity.py` | O registro persistente, o escopo da identidade e a **coerência entre as quatro partes**. | as quatro partes → `Entity` | `Entity`, `EntitySet`, `EntityReference`, `EntityId`, `SemanticMapId`, `EntityProvenance`, `CandidateRejection`, `RejectionReason`, `UnknownEntityError`, `ForeignEntityReferenceError` | Decidir mesmo objeto, materializar, serializar, resumir geometria, alocar identidade. |
| `geometry.py` | O suporte 3D como contrato e a resolução das suas referências contra o mapa. | `GeometryReference` + `GeometrySource` → `EntityGeometry` / pontos | `EntityGeometry`, `SupportStatistics`, `EntityOrientation`, `SpatialSummaryProvenance`, `GeometryDiagnostic(Kind)`, `resolve_geometry`, `EmptyGeometrySupportError`, `GeometryResolutionError` | Escolher limiar, calcular resumo, conhecer a política que o preenche, copiar XYZ para a entidade. |
| `semantic_state.py` | O estado semântico sem colapso, e a ordem canônica que ele exige. | hipóteses + incerteza → `EntitySemanticState` | `EntitySemanticState`, `EntityHypothesis`, `EntityAttribute`, `EntityUncertainty`, `AmbiguityState`, `WithheldPrimaryHypothesis`, `derive_ambiguity_state`, `sorted_attributes` | Eleger vencedor, decidir se existe primária, escalar fator, ler artifact. |
| `evidence.py` | Onde está a evidência que suporta a entidade, só por referência. | `FusedEvidence` + manifest → `EntityEvidenceLinks` | `EntityEvidenceLinks`, `FusedEvidenceRef`, `EntityFeatureRef`, `evidence_links_from_fused_evidence`, `feature_refs_of`, `fusion_artifact_digest` | Copiar imagem, máscara, embedding ou payload de fusão; conferir integridade; abrir run. |
| `temporal.py` | Quando e quantas vezes a entidade foi observada. | `PhysicalObservationGroup` → `EntityTemporalState` | `EntityTemporalState`, `ObservationRef`, `TemporalProvenance`, `EntityLifecycle`, `summarize_temporal_state`, `TemporalEvidenceError` | Velocidade, trajetória, rastreamento, predição, re-identificação; fabricar instante; inferir `stale`. |

## Políticas

| Arquivo | Responsabilidade | Entrada → saída | Conceitos | Não deve possuir |
| --- | --- | --- | --- | --- |
| `geometry_summary.py` | A política versionada do resumo espacial e a sua verificação. | suporte + `GeometrySummaryPolicy` → `EntityGeometry` / problemas | `GEOMETRY_SUMMARY_ALGORITHM_ID`, `GeometrySummaryPolicy`, `OrientationPolicy`, `summarize_geometry`, `verify_geometry_summary` | Implementar a matemática (é de `support_shape`), persistir, decidir identidade de entidade. |
| `state_mapping.py` | A projeção de `FusedEvidence` para o estado e os atributos derivados. | `FusedEvidence` → `EntitySemanticState` | `SEMANTIC_STATE_MAPPING_RULE_ID`, `CLASS_ATTRIBUTE_DERIVATION_ID`, `CLAIM_ATTRIBUTE_DERIVATION_ID`, `semantic_state_from_fused_evidence` | Descartar hipótese, agregar valores de um atributo, decidir a primária, escalar o fator. |
| `primary_hypothesis.py` | Quando uma hipótese pode ser exposta como primária. | hipótese + suporte → primária ou retenção | `PRIMARY_HYPOTHESIS_POLICY_ID`, `MIN_SUPPORT_PRIMARY_HYPOTHESIS_POLICY_ID`, `MinimumSupportPrimaryPolicy`, `withheld_for_lack_of_support` | Contar suporte por conta própria (é de `FusedEvidence`), ponderar por confiança, criar registro de incerteza. |
| `interaction_radius.py` | Converter o fator de interação numa distância, pela extensão medida. | estado + extensão + fator → estado com o atributo | `INTERACTION_RADIUS_DERIVATION_ID`, `EXTENT_RULE_HALF_MAX_EXTENT`, `InteractionRadiusPolicy`, `attach_interaction_radius` | Tratar o raio como medição, alterar hipótese ou incerteza, aparecer na fachada. |
| `materialization.py` | O baseline `one-support-one-entity-v1`: orquestra, aloca identidade e recusa explicitamente. | `FusionOutcome` selecionados → `EntityMaterialization` | `ENTITY_MATERIALIZATION_POLICY_ID`, `ENTITY_ID_POLICY_ID`, `entity_id_for`, `EntityMaterializationPolicy`, `EntityMaterialization`, `MaterializationInputError`, `EntityContractError` | Comparar dois suportes, fundir, dividir, re-identificar, carregar run não selecionado, engolir erro que não seja uma das quatro causas. |

## Auditoria

| Arquivo | Responsabilidade | Entrada → saída | Conceitos | Não deve possuir |
| --- | --- | --- | --- | --- |
| `evidence_validation.py` | Conferir as referências contra os artifacts que elas nomeiam, **relatando**. | entidade + runs de fusão → `EvidenceIntegrityIssue[]` | `EvidenceIntegrityKind` (11 espécies), `EvidenceIntegrityIssue`, `validate_entity_evidence`, `validate_evidence_of_entities` | Levantar no primeiro problema, reparar, reescrever upstream, alcançar política científica. |
| `evidence_trace.py` | Percorrer as cadeias de proveniência, **levantando**. | entidade → traços de identidade | `trace_entity_evidence`, `trace_geometry_sources`, `ContributionTrace`, `EntityEvidenceTrace`, `GeometrySourceTrace`, `EvidenceTraceError` | Devolver payload, carregar runtime de percepção ou modelo, relatar parcialmente. |

## Persistência

| Arquivo | Responsabilidade | Entrada → saída | Conceitos | Não deve possuir |
| --- | --- | --- | --- | --- |
| `serialization.py` | Adaptar os contratos para/de registros JSON, revalidando na volta. | contratos ↔ JSON | codecs explícitos, `ENTITY_SCHEMA_VERSION`, deltas posicionais de geometria | Filesystem I/O, escolher caminho, decidir política, confiar num registro sem revalidar. |
| `run_artifact/layout.py` | O contrato físico: caminhos, versão de schema, erros e formato de bytes. | — → constantes e erros | `SCHEMA_VERSION`, caminhos de `outputs/`/`metrics/`, `encode_line`, `encode_document`, `MappingDebugLevel`, `DEBUG_ENTITY_LIMIT` | Ler ou escrever arquivo, conhecer contrato de domínio, agregar métrica. |
| `run_artifact/manifest.py` | Metadata do run, linhagem explícita e decoding de um manifest já lido. | registro JSON → `SemanticMappingRunManifest` | `MappingRunLineage`, `SemanticMappingRunManifest`, `lineage_from_fusion_manifest`, `load_manifest` | Varrer diretório, verificar integridade, escrever, descobrir upstream. |
| `run_artifact/tables.py` | As tabelas de índice derivadas, uma linha por entidade. | entidade + offset → linhas | `IndexTables`, `encode_rejection` | Validar, medir, escrever em disco, decidir layout físico. |
| `run_artifact/metrics.py` | Checar em fluxo e agregar o que o manifest e `metrics/` afirmam. | fluxo de `Entity` → contagens, distribuições, recusas | `RunTally`, política única do run, linhagem, distribuições por entidade | Guardar linha de `outputs/`, medir qualidade científica, misturar runtime com evidência. |
| `run_artifact/writer.py` | Transmitir as entidades para o disco e publicar o run atomicamente. | fluxo de `Entity` → artifact + manifest | `SemanticMappingRunWriter`, deslocamentos contados na escrita, `AtomicRunDirectory` | Calcular caminho, alocar identidade, manter registro de runs, sobrescrever um run, renderizar debug. |
| `run_artifact/reader.py` | Abrir um run finalizado e ler por identidade, sem carregar o resto. | diretório → entidades, índices, métricas | `SemanticMappingRunReader`, leitura por offset, `verify_integrity`, `validate_references` | Escrever ou republicar, aceitar `__DEBUG__/` como fonte, confiar em registro adulterado. |
| `run_artifact/debug.py` | Escrever evidência humana de diagnóstico. | `Entity` → arquivos em `__DEBUG__/` | `write_entity_debug`, amostragem por nível | Entrar no inventário do manifest, ser dependência de qualquer estágio a jusante, mudar `outputs/`. |
| `__init__.py` | Fachada pública cross-capability. | símbolos internos → API pública | `__all__` explícito | Exportar invariante, digest canônico, matemática de forma, `attach_interaction_radius`, `EntityContractError`, `SCHEMA_VERSION` ou qualquer helper interno. |

## Onde cada validação acontece

A validação fica no boundary que possui a regra, e sempre antes de persistir.

| Etapa | Checagens | Owner | Consequência |
| --- | --- | --- | --- |
| Construção de parte | presença, ordem canônica, frame, faixa numérica, digest do suporte | o próprio contrato (`__post_init__`) | `ValueError` |
| Coerência da entidade | hipótese e incerteza traçam para a evidência vinculada; histórico e vínculos listam os mesmos frames; primária retida dentro das observações; representação 3D dentro do suporte | `entity.py` (`Entity.__post_init__`) | `ValueError` |
| Resolução de geometria | mapa divergente, referência inexistente, frame divergente | `geometry.py` | `GeometryResolutionError` |
| Evidência temporal | nenhuma observação, observação repetida, mais de um domínio de clock | `temporal.py` | `TemporalEvidenceError` |
| Seleção de candidatos | suporte selecionado duas vezes, mapa da linhagem ≠ mapa da geometria | `materialization.py` | `MaterializationInputError` |
| Candidato inválido | suporte vazio, geometria não resolvível, evidência temporal inválida, contrato de entidade | `materialization.py` | `CandidateRejection` (as quatro causas); qualquer outro erro **propaga** |
| Run inteiro | ordem estrita, mapa, linhagem, política única, rejeição duplicada | `run_artifact/metrics.py` | `MappingRunArtifactError` |
| Leitura | schema, inventário, tamanho, hash, registro adulterado | `run_artifact/reader.py` | `MappingRunArtifactError` |
| Referências a montante | artifact ausente, divergente, desatualizado, corrompido, linhagem incompatível, listas divergentes | `evidence_validation.py` | `EvidenceIntegrityIssue` (relatado) |

## Decisões de ownership que esta refatoração registrou

- **`CandidateRejection` é contrato, não detalhe de política.** Ela é gravada em `outputs/rejected-candidates.jsonl` e devolvida pelo reader, então vive em `entity.py` com a entidade de que é contrapartida, não no módulo que a produz. É também o que permite que a persistência não importe a materialização.
- **`resolve_geometry` é adaptação de boundary, não política.** Resolver as referências do próprio contrato contra o mapa não recebe nenhum parâmetro de política, e a camada de auditoria precisa dela para relatar `missing_geometry` sem alcançar a política que resume.
- **A matemática do suporte é mecanismo, não ciência.** `support_shape.py` muda por custo; `geometry_summary.py` muda por decisão científica. Separá-los impede que uma otimização pareça uma mudança de política.
- **Os digests de identidade têm um dono.** A forma canônica (chaves ordenadas, separadores sem espaço) era repetida em quatro módulos; corrigi-la num só deixaria três divergentes.
- **`derive_ambiguity_state` fica com o contrato.** `EntitySemanticState` recusa qualquer `ambiguity_state` diferente do que a regra calcula, então mover a regra para a camada de política faria o contrato importar política.
- **A ordem canônica dos atributos fica com o contrato.** `sorted_attributes` é a ordem que `EntitySemanticState` exige, e duas políticas precisam dela; mantê-la numa delas faria a outra importar uma política só para respeitar um contrato.
- **Auditoria é uma camada, não uma política.** Validação e travessia operam sobre evidência já persistida, então a persistência as alcança (`reader.validate_references()`) e elas não alcançam nenhuma regra científica.

## Como usar este mapa em review

1. O arquivo tocado é dono da responsabilidade que a mudança introduz? Se não, a discussão é de ownership.
2. A mudança criou um import para cima? O teste de layering recusa; se a camada estiver errada, mude a declaração e este documento junto.
3. Um arquivo novo apareceu? Ele precisa de camada declarada no teste e de linha nesta tabela.
4. A mudança é científica ou de custo? Se é de custo, ela deveria estar em `support_shape.py` ou num módulo de mecanismo equivalente, não numa política.
