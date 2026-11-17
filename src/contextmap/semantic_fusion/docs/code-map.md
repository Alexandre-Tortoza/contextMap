# Mapa de Código: semantic_fusion

Este documento registra o ownership dos arquivos da capability. Ele é o contrato arquitetural para revisões futuras: se uma função nova não couber na responsabilidade declarada do arquivo, isso é uma discussão de ownership, não um lugar para encaixá-la. Números de linha não fazem parte do contrato.

A capability tem três camadas: **contratos** (o que a fusão afirma), **políticas** (as regras versionadas que os produzem) e **artifact** (como um run é persistido e reaberto). Nenhuma camada chama a de cima.

A persistência (`serialization.py` e `run_artifact/`) importa apenas vocabulário, contratos e módulos de persistência; nunca políticas. `run_artifact/__init__.py` é somente a fachada dos módulos do próprio package. O teste de layering verifica os arquivos do package recursivamente, recusa a fachada da capability em qualquer import interno e exige que todo módulo e subpackage da raiz tenha camada declarada.

## Vocabulário da capability

| Arquivo | Responsabilidade | Entrada → saída | Conceitos principais | Não deve possuir/fazer |
| --- | --- | --- | --- | --- |
| `identity.py` | Identidades locais ao run, derivadas deterministicamente da evidência a montante. | identidades a montante → identidades da fusão | `FusionSupportId`, `EvidenceContributionId`, `FusedEvidenceId`, `FusedHypothesisId`, `evidence_contribution_id_for()`, `fused_evidence_id_for()` | Registro, alocador, contador global, validação de invariante, regra científica. |
| `invariants.py` | Invariantes que mais de um contrato precisa afirmar. | contrato em construção → erro ou silêncio | `require_present`, `require_canonical`, `require_unit_factor`, `require_identified`, `require_geometry_support`, `time_bounds_of`, `producer_key` | Construir contrato, decidir política, escolher limiar, ter side effect, aparecer na fachada pública. |

## Contratos

| Arquivo | Responsabilidade | Entrada → saída | Conceitos principais | Não deve possuir/fazer |
| --- | --- | --- | --- | --- |
| `supports.py` | O *onde* da fusão e a contrapartida explícita do que ficou fora. | geometria + observações → `FusionSupport` / `ExcludedObservation` | `FusionSupport`, `FusionSupportProvenance`, `ExcludedObservation` | Label, classe, identidade de objeto, coordenada própria, política de sobreposição, I/O. |
| `evidence.py` | Contratos de **evidência**: o que uma vista observou, só por referência. | identidades a montante → contratos de evidência | `EvidenceContribution`, `PhysicalObservationGroup`, `SemanticClaimRef`, `VisualFeatureRef`, `SupportSignal`, `ScoreReference`, `RawSemanticScoreReference`, `ObservationQualityRef`, `PointRepresentationRef`, `EvidenceReference`, `EvidenceChannel`, `ChannelProvenance` | Copiar XYZ, embedding, claim ou componente de qualidade; acumular crença; ponderar; eleger hipótese. |
| `fused_evidence.py` | Contratos de **crença em formação** e as invariantes de correlação entre eles. | evidência de várias vistas → `FusedEvidence` | `FusedEvidence`, `FusedHypothesis`, `HypothesisEvidence`, `EvidenceStance`, `UncertaintyRecord`, `UncertaintyKind`, `FusedAttribute`, `FusedInteractionFactor`, `FusionOutcome`, `uncertainty_key` | Eleger vencedor, hipótese primária, confiança combinada, identidade de entidade, probabilidade. |
| `quality_weighting.py` | O peso derivado da qualidade, inspecionável por componente. | qualidade medida → fatores por componente/contribuição/hipótese | `QualityWeighting`, `ContributionWeight`, `ComponentFactor`, `ComponentTreatment`, `ObservationFactor`, `HypothesisSupport` | Substituir a evidência do baseline, descartar observação fraca, virar confiança semântica, ler claim ou score. |

## Políticas

| Arquivo | Responsabilidade | Entrada → saída | Conceitos principais | Não deve possuir/fazer |
| --- | --- | --- | --- | --- |
| `grouping.py` | Separar frame físico de inferência sob uma seleção explícita de runs. | `SpatialObservation` + runs selecionados → `PhysicalObservationGrouping` | `PHYSICAL_OBSERVATION_GROUPING_POLICY_ID`, contagens de frames/inferências/runs/variantes | Eleger vencedor entre runs, descartar inferência repetida, supor independência estatística, fundir run não selecionado. |
| `support_construction.py` | Decidir *o que* conta como suporte: limiares, exclusão, identidade, bounds e centroide. | observações + `GeometrySource` + política → `FusionSupportBuild` | `GEOMETRY_OVERLAP_SUPPORT_POLICY_ID`, `GeometryOverlapSupportPolicy`, `FusionSupportBuild`, `build_fusion_supports()` | Implementar a medida de sobreposição, afirmar semântica, criar entidade, ler claim ou score. |
| `geometry_overlap.py` | Agrupar conjuntos de geometria pelas suas sobreposições de Jaccard. | conjuntos de `GeometryId` + limiar → componentes conexas | `overlap_components()`, `jaccard_overlap()`, bit sets, union-find, poda por índice invertido | Conhecer `SpatialObservation`, política, limiar científico, identidade de suporte ou provenance. Muda por custo, nunca por ciência. |
| `accumulation.py` | Ponto de entrada do baseline e dono da sua configuração; monta uma contribuição por vista. | suporte + observações + visões derivadas dos produtores selecionados → `FusedEvidence` | `BASELINE_ACCUMULATION_POLICY_ID`, `BaselineAccumulationPolicy`, `accumulate_baseline_evidence()` | Regra de label, stance, incerteza ou canal (delegadas); ponderar por qualidade; chamar inferência. |
| `hypotheses.py` | O que é a mesma hipótese e o que cada claim diz sobre ela. | claims de várias vistas → `FusedHypothesis` | `label_key()`, `ViewClaim`, stance, `CATEGORY_ATTRIBUTE`, fator de interação, atributos observados | Pontuar, ordenar, eleger; ler qualidade de observação; aplicar sinônimo, plural ou taxonomia. |
| `uncertainty.py` | Reportar, nunca resolver, o que a evidência deixou indeciso. | hipóteses + claims → `UncertaintyRecord` | `report_uncertainty()`, contradição, ambiguidade, near-tie, evidência insuficiente | Resolver a indecisão, comparar score, ponderar por qualidade, descartar evidência, possuir o id da política. |
| `channels.py` | Que identidades alimentaram cada canal declarado. | canais ativos + evidência → `ChannelProvenance` | `channel_provenance()`, rótulo `backend_id/model/version` | Ativar canal por disponibilidade de dado, converter, renomear ou combinar canais. |
| `quality_aware.py` | A política opcional: rampas declaradas e o peso derivado da qualidade medida. | suporte + `ObservationQuality` + política → `FusedEvidence` com `weighting` | `QUALITY_AWARE_ACCUMULATION_POLICY_ID`, `QualityAwareAccumulationPolicy`, `QualityRamp`, `QualityInput` | Mudar hipótese, contribuição ou incerteza do baseline; descartar observação fraca; aprender peso. |

## Serialização e artifact

| Arquivo | Responsabilidade | Entrada → saída | Conceitos principais | Não deve possuir/fazer |
| --- | --- | --- | --- | --- |
| `serialization.py` | Adaptar os contratos para/de registros JSON, revalidando na volta. | contratos ↔ JSON | codecs explícitos, deltas posicionais de geometria | Filesystem I/O, escolher caminho, decidir política, confiar num registro sem revalidar. Interno: a fachada não o exporta. |
| `run_artifact/layout.py` | Declarar o contrato físico: caminhos, versão de schema, erros e formato de bytes. | — → constantes e erros | `SCHEMA_VERSION`, caminhos de `outputs/`/`metrics/`, `encode_line`, `encode_document`, `SemanticFusionDebugLevel` | Ler ou escrever arquivo, conhecer contrato de domínio, agregar métrica. |
| `run_artifact/manifest.py` | Contratos de metadata do run e decoding de um registro já parseado. | registro JSON → `SemanticFusionRunManifest` | `FusionRunLineage` (seleção explícita), `SemanticFusionRunManifest`, `load_manifest` | Varrer diretório, verificar integridade, escrever, descobrir upstream. |
| `run_artifact/writer.py` | Transmitir os outcomes para o disco e publicar o run atomicamente. | `FusionOutcome` stream → artifact + manifest | `SemanticFusionRunWriter`, índices derivados, `AtomicRunDirectory` | Calcular caminho, alocar identidade, manter registro, guardar linha de tabela em memória, renderizar debug. |
| `run_artifact/reader.py` | Abrir um run finalizado e ler por identidade, sem carregar o resto. | diretório → suportes, evidência, contribuições | `SemanticFusionRunReader`, leitura por offset, `verify_integrity()` | Escrever ou republicar, aceitar `__DEBUG__/` como fonte, confiar em registro adulterado. |
| `run_artifact/metrics.py` | Checar e agregar em fluxo o que o manifest e `metrics/` precisam. | `FusionOutcome` stream → contagens, distribuições, recusas | `RunTally`, política única do run, linhagem, distribuições por suporte | Guardar linha de `outputs/`, medir qualidade científica, misturar runtime com evidência. |
| `run_artifact/debug.py` | Escrever evidência humana de diagnóstico. | `FusionOutcome` → arquivos em `__DEBUG__/` | `write_support_debug()`, amostragem por nível | Entrar no inventário do manifest, ser dependência de qualquer estágio a jusante, mudar `outputs/`. |
| `__init__.py` | Fachada pública cross-capability. | símbolos internos → API pública | `__all__` explícito | Exportar codec, motor de sobreposição, invariante, `ViewClaim` ou qualquer helper interno. |

## Onde cada validação acontece

A validação fica no boundary que possui a regra, e sempre antes de persistir.

| Etapa | Checagens | Owner | Consequência |
| --- | --- | --- | --- |
| Construção de contrato | presença, ordem canônica, unidade, frame, faixa numérica | o próprio contrato (`__post_init__`) | `ValueError` |
| Correlação interna | grupo, hipótese, atributo, fator e incerteza resolvem para contribuições e claims reais | `fused_evidence.py` | `ValueError` |
| Seleção de runs | artifact regional não selecionado, sequência divergente, associação regional ambígua | `grouping.py` | `ValueError` |
| Elegibilidade e mapa | geometria mínima, mapa divergente, observação repetida | `support_construction.py` | `ValueError` ou `ExcludedObservation` |
| Canal declarado | canal sem entrada, dado de canal não declarado | `accumulation.py` | `ValueError` ou dado ignorado |
| Definições de qualidade | qualidade medida sob outra versão | `quality_aware.py` | `ValueError`, antes de qualquer peso |
| Run inteiro | política única, linhagem, ordem estrita dos suportes | `run_artifact/metrics.py` | `FusionRunArtifactError` |
| Leitura | schema, inventário, tamanho, hash, registro adulterado | `run_artifact/reader.py` | `FusionRunArtifactError` |

## Decisões de ownership que esta refatoração registrou

- **`ExcludedObservation` é contrato, não detalhe de política.** Ela é persistida em `outputs/` e devolvida pelo reader, então vive em `supports.py` com o suporte que ela complementa, não no módulo que a produz.
- **`FusionOutcome` é contrato de domínio, não de artifact.** Ele é o par que Semantic Mapping consome, então vive em `fused_evidence.py`; o artifact apenas o persiste.
- **A medida de sobreposição é mecanismo, não ciência.** `geometry_overlap.py` muda por custo; `support_construction.py` muda por decisão científica. Separá-los impede que uma otimização pareça uma mudança de política.
- **As invariantes são vocabulário da capability, não da `shared`.** Elas só fazem sentido com a semântica desta capability (ordem canônica da evidência, fator de fusão, domínio de clock de aquisição), então ficam aqui e não aparecem na fachada.
- **A regra de incerteza não possui o id da política.** `report_uncertainty()` recebe `policy_id`: a política é dona da sua própria identidade versionada, e as regras só a citam.
