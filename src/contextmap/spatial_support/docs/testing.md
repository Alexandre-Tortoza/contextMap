# Testes de Spatial Support

## Responsabilidade → teste

| Responsabilidade | Teste |
| --- | --- |
| Invariantes do vocabulário (`models.py`) | `tests/spatial_support/test_semantic_spatial_support_contract.py` |
| A junção e as suas duas recusas (`service.py`) | `tests/spatial_support/test_semantic_spatial_support_join.py` |
| Identidade da política e o que um run registra dela | `tests/spatial_support/test_entity_voxel_support_policy.py` |
| Round-trip, inventário e recusas do run (`run_artifact/`) | `tests/spatial_support/test_semantic_spatial_support_artifact.py` |
| Derivação sobre três artifacts reais (`derivation.py`) | `tests/end_to_end/test_semantic_spatial_support_derivation.py` |
| O caminho de produção pela CLI | `tests/end_to_end/test_spatial_support_cli_derivation.py` |
| Camadas internas e o monopólio de `derivation.py` sobre readers estrangeiros | `tests/architecture/test_spatial_support_layering.py` |

### Fora do diretório da capability

| Teste | O que cobre daqui |
| --- | --- |
| `tests/architecture/test_boundaries.py` | A capability é declarada e só pode importar `shared`, `ingestion`, `geometric_mapping`, `artifact`; nenhum import de submódulo cross-capability |
| `tests/architecture/test_runtime_boundaries.py` | A CLI só importa a raiz pública da capability dentro de `_derive_spatial_support`; o mesmo import em qualquer outro ponto é violação |

Os testes reaproveitam os builders de `tests/geometric_mapping` (`map_builders`,
`voxel_builders`), importáveis por nome porque `pyproject.toml` põe esse diretório no
`pythonpath`. É uma dependência real: mudar os builders da agregação pode quebrar os testes
daqui.

## Níveis

- **unit**: os invariantes de `models.py` e o `fingerprint()` da política — nada em disco.
- **contract**: a junção, rodada sobre uma agregação voxel **real** de um mapa sintético com
  revisita, não sobre um duplo. É onde o acoplamento com `membership()` é verificado.
- **integration**: escrita e leitura de um run em `tmp_path`, com inventário e adulteração.
- **architecture**: camadas internas, fronteiras da capability e a exceção de import da CLI.
- **end-to-end**: a derivação sobre uma cadeia sintética completa, e o comando de CLI.

Não há benchmark nem avaliação científica: a capability não escolhe modelo, não pontua e não
combina sinais heterogêneos.

## Invariantes que nunca podem quebrar

1. `cell_id` é função pura de `(grid_fingerprint, key)` e **nunca** codifica um índice de
   ponto bruto — é a razão de a capability existir.
2. Toda célula é expressa no frame declarado pelo suporte.
3. Uma célula só é emitida quando alguma entidade a suporta.
4. O suporte de uma célula é ordenado por `entity_id` e único.
5. `support_weight` é ao menos um, e é contagem de medições: nunca probabilidade, nunca score.
6. Uma entidade só pode referenciar geometria de um único mapa.
7. Uma referência de outro mapa é recusada, com o id na mensagem.
8. Um ponto que não cai em voxel nenhum é recusado, nunca ignorado.
9. Os três artifacts a montante compartilham um `map_id`, e tanto o `ContextMap` quanto a
   agregação citam exatamente o digest do mapa bruto fornecido; a recusa não deixa diretório
   de saída para trás.
10. Uma adulteração de mesmo tamanho no payload do `ContextMap` é detectada antes da leitura
    das entidades.
11. Falhas públicas dos readers a montante são expostas como `SpatialSupportArtifactError`,
    para a CLI responder com código 1 e sem traceback.
12. Um diretório de run existente nunca é substituído.
13. Um `schema_version` que o pacote não entende é recusado ao abrir, inclusive o `0.1.0`
    anterior: durante `v0.x` não há camada de migração.
14. O round-trip é exato: ler um run reproduz o suporte escrito, célula por célula.
15. Um `outputs/cells.jsonl` adulterado é reportado pelo inventário.
16. As células saem ordenadas por chave de voxel — garantia emprestada de `VoxelAggregation`,
    fixada aqui para falhar alto se mudar lá.
17. Um run registra o `fingerprint` da política em `config.json` e no manifest.
18. Todo documento JSON contratual termina em newline; os bytes entram no inventário, então
    mudar o rendering é um bump de `SCHEMA_VERSION`.

## Regressões protegidas

Os testes end-to-end impedem duas regressões de provenance: reinterpretar um `ContextMap`
sobre outro mapa bruto que reutiliza o mesmo `map_id`, e consumir um payload adulterado que
preserva o tamanho inventariado. O teste da CLI fixa que um upstream ausente produz a saída
controlada documentada, sem exceção não tratada.

## Como rodar

```bash
python -m pytest tests/spatial_support \
                 tests/end_to_end/test_semantic_spatial_support_derivation.py \
                 tests/end_to_end/test_spatial_support_cli_derivation.py \
                 tests/architecture
```

O gate canônico é `make check` (ruff, ruff format, mypy, pytest).

## Mudanças estruturais

Uma reorganização que não muda comportamento não inventa um `Red` artificial: os contratos já
estão protegidos pelos testes acima. Em troca, uma mudança de bytes persistidos deve ser
demonstrada, não afirmada — escreva o mesmo suporte antes e depois e compare o inventário de
arquivos. Foi assim que cada extração desta capability foi verificada, e é assim que se vê que
um bump de schema mexeu só nos documentos JSON e deixou `outputs/cells.jsonl` idêntico.

Um módulo novo precisa de camada declarada em
`tests/architecture/test_spatial_support_layering.py` **e** de linha no
[code-map](code-map.md); sem as duas, o teste de camadas falha.
