# Mapa de código de Spatial Support

Este documento responde a uma pergunta: **qual arquivo devo abrir para entender ou alterar
esta responsabilidade?** Ele descreve *ownership*, não o conteúdo atual — tamanho de arquivo é
um sinal, não o critério de divisão. A coluna "Não deve possuir" é o que faz dele um contrato
de review: se uma função nova não couber na responsabilidade declarada, a discussão é de
ownership, não de onde é mais prático colar o código.

## Raiz da capability

Camada interna: `models`, `policy` ← `service` ← `derivation`; e
`models`, `policy` ← `run_artifact` ← `derivation`.

| Arquivo | Responsabilidade | Entrada → saída | Conceitos | Não deve possuir |
| --- | --- | --- | --- | --- |
| `__init__.py` (52) | Fachada cross-capability: quinze nomes, `__all__` explícito | — | A superfície pública | Lógica; import de dentro da capability para ela mesma |
| `models.py` (170) | O vocabulário do suporte e os seus invariantes de construção | — → tipos congelados | `SpatialCellId`, `spatial_cell_id_for`, `EntityCellInput`, `EntitySupport`, `SpatialCell`, `SemanticSpatialSupport` | Qualquer import de dentro da capability (é folha); I/O; a regra de junção; serialização |
| `policy.py` (67) | A identidade versionada da regra: o que a junção computa e a semântica do peso | — → documento e `fingerprint()` | `EntityVoxelSupportPolicy`, `ENTITY_VOXEL_SUPPORT_POLICY_{ID,VERSION}` | A grade (é da agregação); cálculo; leitura de configuração |
| `service.py` (121) | A junção: atribuir cada ponto bruto da entidade ao voxel que o resume | `EntityCellInput[]` + `VoxelAggregation` + `GeometryBlockSource` → `SemanticSpatialSupport` | `build_semantic_spatial_support`, `SemanticSpatialSupportError`, `IMPLEMENTED_POLICY` | Re-binning próprio; persistência; abrir artifact; leitura de configuração |
| `derivation.py` (128) | Seleção dos três upstreams, verificação de identidade e hashes, normalização dos erros públicos e montagem da linhagem | três diretórios → `SpatialSupportManifest` | `derive_semantic_spatial_support_artifact` | A regra de junção; o layout em disco; interpretar a linhagem que grava |

## `run_artifact/` — o run persistido

Camada interna: `layout` ← `manifest` ← {`writer`, `reader`}.

| Arquivo | Responsabilidade | Entrada → saída | Conceitos | Não deve possuir |
| --- | --- | --- | --- | --- |
| `__init__.py` (41) | Fachada **interna** do run; declara a camada no docstring | — | Seis nomes do run | Reexportar algo fora de `run_artifact/`; ser importada de fora da capability |
| `layout.py` (158) | O contrato físico: onde cada arquivo vive e como os seus bytes são formados | `SpatialCell` ↔ registro JSON | `SCHEMA_VERSION`, os sete caminhos, `SpatialSupportArtifactError`, `cell_record`, `cell_of` | `AtomicRunDirectory`; a regra de junção; abrir artifact; decidir métrica |
| `manifest.py` (128) | O contrato de metadata e a recusa de um schema desconhecido | `manifest.json` → `SpatialSupportManifest` | `SpatialSupportRunId`, `SpatialSupportManifest`, `load_manifest`, `environment_record` | Escrever; conhecer `AtomicRunDirectory`; camada de migração |
| `writer.py` (159) | Publicar atomicamente um suporte já construído, com a política recebida | `SemanticSpatialSupport` + política + linhagem → sete arquivos | `SpatialSupportArtifactWriter` | Derivar; abrir artifact a montante; importar `service`; interpretar a linhagem |
| `reader.py` (101) | Abrir um run publicado, reconstruir o suporte e confrontar o inventário | diretório → `SemanticSpatialSupport`, problemas | `SpatialSupportArtifactReader`, `verify_integrity` | Escrever; reparar um run; conhecer a regra de junção |

## Fora da capability

| Arquivo | O que lhe pertence daqui |
| --- | --- |
| `src/contextmap/runtime/cli.py` | O comando `derive-spatial-support`: parser e handler. Importa a raiz pública **dentro da função**, e só ali |
| `experiments/semantic-voxel-color-20260930/derive_voxel_support.py` | Só a agregação voxel de um mapa real, mais o comando a rodar em seguida |

## Como usar este mapa em review

1. a função nova cabe na responsabilidade declarada do arquivo, ou é uma responsabilidade
   nova que precisa de dono?
2. algum módulo fora de `derivation.py` passou a importar um reader de artifact de outra
   capability?
3. algum módulo de `run_artifact/` passou a importar `service` ou `derivation`?
4. `models.py` continua folha?
5. um nome novo entrou em `__all__` com um consumidor real fora da capability?
6. um módulo novo apareceu sem camada declarada em
   `tests/architecture/test_spatial_support_layering.py` e aqui?
7. o layout em disco mudou sem subir `SCHEMA_VERSION`?
