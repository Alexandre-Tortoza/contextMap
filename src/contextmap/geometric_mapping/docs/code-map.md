# Mapa de Código: geometric_mapping

Este documento registra o ownership dos arquivos da capability. O objetivo é tornar explícitos responsabilidade, fluxo de dados, conceitos principais e limites de cada módulo. Os números de linhas não fazem parte do contrato porque mudam com frequência.

## Módulos core

| Arquivo | Responsabilidade | Entrada → Saída | Conceitos principais | Não deve possuir/fazer |
| --- | --- | --- | --- | --- |
| `models.py` | Contratos canônicos de geometria persistente e identidade espacial. | Tipos primitivos → contratos de domínio | `GeometryPoint`, `GeometricMap`, `Bounds3D`, `GeometryReference`, lineage de transform | I/O, leitura de artifact, política de pose, deskew, agregação voxel ou semântica. |
| `ports.py` | Fronteira pública de leitura para consumidores de outras capabilities. | Referências/limites → geometria consultável | `GeometrySource` | Conhecer formato de arquivo, implementação de índice, reader concreto ou detalhes de backend. |
| `contracts.py` | Agrupar os contratos do planejamento que separam fontes de evidência das decisões de processamento. | IDs, frames, seleção e políticas → `GeometryMappingSources` / `GeometryMappingPolicy` | fontes, políticas, lineage declarada | Resolver seleção, ler artifact, transformar pontos ou persistir dados. |
| `inputs.py` | Montagem e validação do plano de geometric mapping. | `SequenceSelectionResult` + calibração + trajetória + políticas → `GeometryMappingPlan` | `GeometryMappingPlan`, `PreparedGeometryScan`, `RejectedGeometryScan`, `PointCloudLayout`, motivos de rejeição | Aplicar transformação para map frame, acumular mapa, escrever artifacts ou inferir semântica. |
| `transformation.py` | Transformar scans preparados para o frame do mapa com trace auditável. | `GeometryMappingPlan` / `PreparedGeometryScan` → `MapFrameScan` | `MapFrameScan`, `TransformTrace`, `TracedTransform`, verificação de transform | Selecionar observações, resolver ownership de artifact, acumular mapa ou alterar política de input. |
| `accumulation.py` | Acumular scans transformados em fluxo e produzir o resumo de build do mapa. | `MapFrameScan` stream → `GeometryBuildSummary` | `MapAccumulator`, `GeometryBuildSummary`, `ScanVoxelPolicy`, `ScanRecord` | Resolver pose/calibração, fazer deskew, interpretar semântica ou possuir o formato de artifact completo. |
| `geometry_storage.py` | Formato binário empacotado e acesso por referência/índice à geometria persistida. | coordenadas + índices de origem → `PackedGeometry` | `PackedGeometry`, layout binário, acesso aleatório, estatísticas | Definir política de mapeamento, transformar scans, validar trajetória ou montar artifact de run. |
| `serialization.py` | Adaptar contratos de domínio para/de registros JSON. | contratos de domínio ↔ JSON | codecs explícitos e versionáveis | Fazer filesystem I/O, acessar ROS/SDKs, executar lógica de mapping ou decidir política. |
| `motion_correction.py` | Representar estado, evidência e política de correção de movimento. | observação + declaração/política → decisão de correção | `MotionCorrectionRecord`, `MotionCorrectionPolicy`, `MotionCorrectionEvidence`, `ScanDisposition` | Implementar transformação sensor→mapa, persistir artifact ou classificar conteúdo da cena. |
| `deskew.py` | Implementar a correção por interpolação de pose para pontos dentro de um scan. | scan + trajetória + política temporal → scan corrigido | `PoseInterpolationDeskewPolicy`, `PointTimeUnit`, `PointTimeReference` | Escolher scans, acumular mapa, escrever artifact ou esconder ausência de informação temporal. |
| `bulk_geometry.py` | Leitura vetorizada de blocos de geometria. | `PackedGeometry` → `GeometryBlock` | `GeometryBlock`, `GeometryBlockSource`, chunking | Alterar o armazenamento autoritativo, definir política de mapping ou anexar semântica. |

## Representação derivada opcional: `voxel_aggregation/`

A agregação voxel inter-scan é derivada, opcional e não canônica. Ela preserva lineage para o mapa bruto e não substitui a geometria autoritativa.

| Arquivo | Responsabilidade | Entrada → Saída | Conceitos principais | Não deve possuir/fazer |
| --- | --- | --- | --- | --- |
| `grid.py` | Definir identidade e geometria da grade voxel. | frame + origem + resolução → `VoxelGridSpec` | grade, resolução, voxel key | Ler artifact, acumular observações ou decidir política semântica. |
| `policy.py` | Definir a política e a identidade versionada da agregação inter-scan. | configuração → `InterScanVoxelPolicy` | versão, política, parâmetros | Executar agregação ou persistir artifact. |
| `aggregate.py` | Calcular agregados voxel preservando contagens e lineage recuperável. | geometria bruta + grid/policy → `VoxelAggregation` | `InterScanVoxelAggregator`, `VoxelAggregate`, `VoxelContribution` | Modificar o mapa bruto, tornar agregados canônicos ou esconder contribuições de origem. |
| `artifact.py` | Persistir e reabrir o artifact derivado de agregação. | `VoxelAggregation` ↔ artifact derivado | `VoxelAggregationManifest`, reader/writer, identidade do mapa fonte | Tornar o artifact derivado fonte autoritativa ou reescrever o `GeometricMapArtifact` bruto. |
| `__init__.py` | Re-exportar a superfície pública da representação derivada. | símbolos internos → API pública | fachada do subpackage | Implementar regra de negócio ou criar dependências laterais entre módulos. |

## Artifact de geometric mapping: `run_artifact/`

O artifact do run é imutável e autodescritivo. Schema, leitura e escrita ficam separados.

| Arquivo | Responsabilidade | Entrada → Saída | Conceitos principais | Não deve possuir/fazer |
| --- | --- | --- | --- | --- |
| `manifest.py` | Definir schema/tipos do manifest e fazer decoding puro de registros já parseados. | registro JSON parseado → `GeometricMapArtifactManifest` | schema version, run/map IDs, `decode_manifest` | Ler filesystem, escrever arquivos ou depender do reader. |
| `reader.py` | Abrir artifacts persistidos, verificar integridade e reconstruir contratos. | diretório do run → reader + manifest + geometria | inventário, hashes, `GeometricMapArtifactReader` | Criar/republicar artifacts ou possuir lógica de mapping. |
| `writer.py` | Orquestrar transformação/acumulação e publicar o run atomicamente. | `GeometryMappingPlan` + política de agregação → artifact + manifest | `GeometricMapArtifactWriter`, `AtomicRunDirectory`, lineage/config/metrics | Reabrir o próprio artifact via helper privado do reader, inferir inputs ou alterar evidência fonte. |
| `__init__.py` | Re-exportar os contratos públicos do artifact. | símbolos internos → API pública | fachada do artifact | Implementar I/O ou regra de domínio. |

## Pipeline de validação

A validação acontece no owner mais próximo do dado e antes de persistência.

| Etapa | Checagens | Owner | Consequência |
| --- | --- | --- | --- |
| Observação | tipo, frame, clock, payload | `inputs.py` | rejeição tipada ou erro |
| Layout | shape, point count, dtype | `inputs.py` | rejeição tipada ou erro |
| Pose lookup | resolução e limites temporais | `inputs.py` | rejeição tipada ou erro |
| Transform estático | conectividade e calibração | `inputs.py` / `transformation.py` conforme o contrato | rejeição ou `GeometryTransformError` |
| Coordenadas | finitude e validade antes de persistir | `transformation.py` | erro ou ponto descartado conforme contrato |
| Acumulação | consistência de bounds, contagens e índices | `accumulation.py` | `AccumulationError` |
| Artifact | schema, inventário, tamanho e hash | `run_artifact/` | publicação/leitura falha explicitamente |

## Impacto desta refatoração na API pública

A capability continua sendo consumida pela fachada `contextmap.geometric_mapping`, mas esta refatoração altera intencionalmente parte dos símbolos públicos de planejamento, transformação e resultado de build. Como o projeto está em v0.x, não há shim de compatibilidade para os nomes removidos.

Principais substituições:

| Antes | Agora |
| --- | --- |
| `GeometryInputPlan` | `GeometryMappingPlan` |
| `GeometryInput` | `PreparedGeometryScan` |
| `GeometryInputRejection` | `RejectedGeometryScan` |
| `assemble_geometry_inputs()` | `build_geometry_mapping_plan()` |
| `assemble_geometry_inputs_from_artifacts()` | `build_geometry_mapping_plan_from_artifacts()` |
| `TransformedScan` | `MapFrameScan` |
| `transform_scan()` | `transform_scan_to_map_frame()` |
| `transform_scans()` | `transform_scans_to_map_frame()` |
| `AccumulatedMap` | `GeometryBuildSummary` |

Contratos cross-capability que continuam estáveis, como `GeometrySource`, `GeometryPoint`, `GeometryReference` e `GeometricMap`, permanecem disponíveis pela fachada pública.

## Dependências e limites

A direção conceitual é:

```text
models / contracts
        ↓
inputs
        ↓
transformation
        ↓
accumulation + geometry_storage
        ↓
run_artifact
        ↓
voxel_aggregation  (derivado e opcional)
```

Consumidores externos devem depender de `contextmap.geometric_mapping` e de seus ports/contratos públicos, não de módulos internos concretos.

## Quality gates

- Ruff e format.
- mypy.
- suíte de testes unitários, de contrato, integração e arquitetura.
- round-trip e integridade dos artifacts.
- avaliação científica fora do core quando a mudança altera qualidade geométrica.

## Referências

- [`README.md`](README.md) - responsabilidade e contratos públicos.
- [`architecture.md`](architecture.md) - direção de dependências e boundaries.
- [`data-flow.md`](data-flow.md) - fluxo end-to-end.
- [`testing.md`](testing.md) - estratégia de testes.
- [`contracts.md`](contracts.md) - contratos detalhados.
- [`inputs.md`](inputs.md) - montagem e validação.
- [`transformation.md`](transformation.md) - transform e lineage.
- [`accumulation.md`](accumulation.md) - acumulação e índices.
- [`spatial-access.md`](spatial-access.md) - acesso por `GeometrySource`.
- [`inter-scan-aggregation.md`](inter-scan-aggregation.md) - representação voxel derivada.
- [`artifact.md`](artifact.md) - artifact persistido do run.
