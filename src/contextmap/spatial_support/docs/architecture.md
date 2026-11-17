# Arquitetura de Spatial Support

## Ownership

A capability possui um conceito e uma regra:

- o **endereço espacial** (`SpatialCell`, `SpatialCellId`, `spatial_cell_id_for`) e o suporte
  que se acumula nele (`EntitySupport`, `SemanticSpatialSupport`);
- a **atribuição** de cada ponto bruto de uma entidade ao voxel em que ele cai
  (`build_semantic_spatial_support`), com a identidade versionada dessa regra
  (`EntityVoxelSupportPolicy`).

Ela **não** possui: a entidade (é de `artifact`, via `ContextEntityId`), a grade nem a
agregação voxel (são de `geometric_mapping`), o frame (é de `ingestion`, consumido só como
tipo), e nenhuma política de hipótese semântica — um voxel acumula suporte e nunca vira
entidade.

O `grid_fingerprint` e o `aggregation_rule` que aparecem em todo `SpatialCellId` e no manifest
são **emprestados** da agregação, não definidos aqui. É deliberado: a identidade da célula
precisa mudar quando a grade muda, e quem define a grade é quem a possui.

## Direção de dependências

```text
ContextMapArtifact   VoxelAggregationArtifact   GeometricMapArtifact
          \                   |                        /
           \                  v                       /
            `--------> derivation.py <---------------'
                         |        |
                         v        v
                    service   run_artifact
                         \        /
                          v      v
                      models, policy
```

### Dependências permitidas

`shared`, `ingestion`, `geometric_mapping`, `artifact` — exatamente o que
`tests/architecture/test_boundaries.py` declara. `ingestion` entra só pela identidade de
frame (`FrameId`), como tipo; nenhuma mensagem, nenhum leitor de sequência.

### Dependências proibidas

Todo o resto, incluindo `runtime`. Nada em `src/contextmap/` depende de `spatial_support`, e é
isso que garante que não há ciclo com `artifact`, de quem esta capability depende.

## Camadas internas

```text
models, policy  ←  service      ←  derivation
models, policy  ←  run_artifact/layout ← manifest ← {writer, reader}  ←  derivation
```

Duas regras sustentam a separação, e as duas são verificadas por
`tests/architecture/test_spatial_support_layering.py`:

1. **`run_artifact/` nunca importa a regra.** O writer recebe a política como parâmetro e
   registra-a; ele não a implementa. Sem isso, o formato persistido volta a depender da regra
   que o alimenta.
2. **`derivation.py` é o único módulo que abre artifact de outra capability.**
   `test_boundaries.py` permite `artifact` e `geometric_mapping` para a capability inteira,
   então ele nunca pegaria um `ContextMapArtifactReader` reaparecendo no writer — que é
   exatamente o estado de que esta capability saiu.

A fachada `__init__.py` alcança todas as camadas, e por isso é proibida de dentro: de fora ela
é a única porta; de dentro, o import é sempre direto ao módulo dono.

## API pública

Quinze nomes, em `contextmap.spatial_support`. O subpacote `run_artifact/` é interno: nenhum
consumidor fora da capability pode importá-lo, e `test_boundaries.py` recusa qualquer import de
submódulo cross-capability. A superfície está listada no [README](README.md#superfície-pública).

## Adapters e backends

Não há. A capability tem um formato persistido, uma regra de junção, zero ponto de variação
real e nenhuma dependência externa além da biblioteca padrão (`json`, `platform`). Criar um
`Protocol`, um `backends/` ou uma factory aqui seria estrutura sem pressão de implementação
(AGENTS.md §18, §31).

## Composição e runtime

A capability **não é estágio do DAG**: nenhum estágio canônico consome o seu artifact, e
`src/contextmap/runtime/{catalog,composition,pipeline,config}.py` não a mencionam. O caminho de
produção suportado é o comando `contextmap derive-spatial-support`, cujo handler importa a
raiz pública da capability dentro da própria função — uma exceção registrada em
`tests/architecture/test_runtime_boundaries.py`, de modo que o mesmo import em qualquer outro
ponto da CLI continua sendo violação. Ver [`runtime/docs/cli.md`](../../runtime/docs/cli.md).

## Evidence, belief e knowledge

- **Evidence**: o ponto bruto, com o seu índice, preservado como proveniência da observação.
- **Endereço**: o voxel. Não é evidência nem crença: é o lugar onde evidências se acumulam, e
  a sua identidade é função pura da grade.
- **Belief**: a entidade, que é **referenciada** por `ContextEntityId` e nunca substituída nem
  reescrita. O suporte é evidência de nível de crença *sobre onde* ela está.

`support_weight` é contagem de medições. Não é probabilidade, não é score, e não é combinada
com nenhum dos dois: a semântica do peso entra no `fingerprint()` da política, então mudá-la
obriga uma versão nova em vez de passar silenciosamente.

## Invariantes arquiteturais

- a identidade de uma célula nunca codifica um índice de ponto bruto;
- toda célula é expressa no frame do suporte;
- uma célula só é emitida quando alguma entidade a suporta;
- o suporte de uma célula é ordenado por entidade e único;
- um ponto que não cai em voxel nenhum é recusado, nunca ignorado;
- os três artifacts precisam nomear o mesmo `map_id`, e o `ContextMap` e a agregação precisam
  estar presos ao digest exato do mapa bruto fornecido;
- todos os hashes do `ContextMap` são verificados antes de suas entidades serem consumidas;
- um run finalizado é imutável, e um diretório existente nunca é substituído;
- um run de schema anterior é recusado ao abrir, sem camada de migração.

## Dívida arquitetural declarada

- **A ordenação das células é emprestada.** A junção itera os agregados tocados por índice
  crescente, e isso só é ordem de chave de voxel porque `VoxelAggregation` garante agregados
  ordenados por `(x, y, z)`. A garantia não é nossa; um teste a fixa
  (`test_the_cells_are_ordered_by_voxel_key`) e falha alto se `geometric_mapping` reordenar.
- **`derivation.py` é discutivelmente do runtime.** Selecionar artifact a montante está na
  lista de responsabilidades do composition root (AGENTS.md §19). Ele fica aqui porque duas
  das quatro coisas que faz são de domínio — a recusa de upstream errado e a adaptação de
  `Entity` para `EntityCellInput`. Isolá-lo num módulo próprio torna a mudança futura um
  `git mv` de um arquivo.
- **A agregação voxel a montante não tem produtor no pipeline.** `VoxelAggregationArtifact`
  também não é estágio nem comando, então produzi-la continua fora do runtime. É de
  `geometric_mapping`, não desta capability.
- **A política não tem parâmetro.** `EntityVoxelSupportPolicy` é uma dataclass sem campos:
  versiona o que a junção computa, não sobre o que. O primeiro parâmetro real (por exemplo um
  peso mínimo de suporte) é o que justifica campos.
- **O contrato de transporte não tem teste do lado do consumidor.** O `aggregate_index` existe
  para o viewer, que vive fora deste repositório; nada aqui verifica que ele o usa como o
  contrato diz.
