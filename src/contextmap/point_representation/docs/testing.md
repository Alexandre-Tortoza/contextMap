# Testes de Point Representation

Este documento mapeia responsabilidade → teste, e declara as invariantes que nunca podem quebrar.
Os testes vivem em `tests/point_representation/`, mais os guards de runtime, end-to-end e
arquitetura listados abaixo.

## Responsabilidade → teste

| Responsabilidade | Arquivo | Testes |
| --- | ---: | ---: |
| contratos canônicos, validação, fingerprint e comparabilidade de espaço | `tests/point_representation/test_point_representation_models.py` | 69 |
| extração de suporte: raio, k vizinhos, ordem, preparação de coordenadas | `tests/point_representation/test_support_extraction.py` | 82 |
| política versionada de seleção de centros | `tests/point_representation/test_center_selection.py` | 6 |
| execução independente de backend, falhas explícitas, métricas | `tests/point_representation/test_representation_service.py` | 21 |
| round-trip dos contratos em JSON primitivo | `tests/point_representation/test_point_representation_serialization.py` | 8 |
| run artifact: layout, manifest, linhagem, inventário, leitura por identidade, integridade | `tests/point_representation/test_point_representation_run_artifact.py` | 49 |
| descritor determinístico: definição das features, comportamento numérico | `tests/point_representation/test_geometric_descriptor.py` | 49 |
| fronteira do adapter PTv3 com runtime falso | `tests/point_representation/test_ptv3_backend.py` | 47 |
| runtime Pointcept: voxelização, pooling, checkpoint, falhas de device, sem CUDA | `tests/point_representation/test_ptv3_pointcept_runtime.py` | 40 |
| harness de ablação | `tests/point_representation/test_point_representation_evaluation.py` | 34 |
| PTv3 real em GPU | `tests/point_representation/test_ptv3_pointcept_real.py` | 10 (opt-in) |

Helpers, sem teste próprio: `pointrep_builders.py`, `pointrep_fakes.py`, `pointrep_ptv3_fakes.py`
(cujo `FakePTv3Runtime` é reusado pelos testes de runtime e end-to-end). `tests/point_representation`
está no `pythonpath` do pytest, então são importáveis por nome de módulo.

### Fora do diretório da capability

| Responsabilidade | Arquivo | Testes |
| --- | ---: | ---: |
| composição do estágio: só quando habilitado, com identidade de código, sem fallback entre encoders | `tests/runtime/test_runtime_point_representation.py` | 9 |
| CLI instalado executa o estágio opt-in; nada a jusante o consome | `tests/end_to_end/test_point_representation_stage.py` | 5 |
| leitura do run por outra capability, e recusa de espaços incompatíveis | `tests/entity_resolution/test_resolution_representation_comparison.py` | 25 |

Também protegem a capability, sem serem dela: `tests/runtime/test_runtime_catalog.py` (identidade da
política, o canal recusado continua existindo, estágio opcional desligado por padrão),
`tests/runtime/test_runtime_pipeline.py` (o ramo opcional entra só quando selecionado, e é saída
derivada), `tests/runtime/test_runtime_config.py` (o canal de evidência é recusado mesmo com o
estágio ligado), `tests/runtime/test_runtime_context_build.py` (uma representação que ninguém consome
não é fornecida nem recomputada), `tests/packaging/test_release_scope.py` (fixa
`run_artifact.SCHEMA_VERSION`) e `tests/artifact/test_context_map_lineage.py`.

## Níveis

| Nível | Onde | O que protege |
| --- | --- | --- |
| unit | `test_point_representation_models.py`, `test_support_extraction.py`, `test_center_selection.py`, `test_geometric_descriptor.py` | comportamento determinístico local e validação de invariante no boundary que a possui |
| contract | `test_ptv3_backend.py`, `test_representation_service.py` | que implementações da porta `PointEncoder` e do `PTv3Runtime` respeitem as mesmas invariantes públicas |
| integration | `test_point_representation_run_artifact.py`, `tests/entity_resolution/test_resolution_representation_comparison.py` | escrita/leitura do artifact e o consumo dele por outra capability |
| architecture | `tests/architecture/test_boundaries.py`, `tests/architecture/test_runtime_boundaries.py` | dependências permitidas, e que backend concreto só é importado dentro da factory que o usa |
| end-to-end | `tests/end_to_end/test_point_representation_stage.py` | o CLI instalado, com o estágio ligado por configuração |
| benchmark / evaluation | `test_point_representation_evaluation.py`, `test_ptv3_pointcept_real.py` | qualidade científica e custo real, com hardware |

## Invariantes que nunca podem quebrar

1. **Um run artifact abre com a biblioteca padrão.** `test_point_representation_run_artifact.py`
   importa o reader num subprocesso e falha se numpy ou torch tiverem sido carregados. É o que
   permite auditar um run numa máquina sem GPU e sem o ambiente de modelo.
2. **Importar um backend não carrega biblioteca de modelo.** `test_ptv3_backend.py` e
   `test_ptv3_pointcept_runtime.py` afirmam que nenhum de numpy, torch, spconv, torch_scatter e
   pointcept entra em `sys.modules`. É o que mantém o import de `contextmap` barato e o que torna a
   separação entre adapter e runtime verificável em vez de declarada.
3. **Dimensionalidade igual não torna dois vetores comparáveis.** Só o mesmo fingerprint de
   `RepresentationSpace`. Os testes exigem a recusa entre descritor e PTv3 de mesma dimensão, e entre
   o mesmo encoder sob políticas de suporte diferentes.
4. **A ordem do suporte não depende da ordem da fonte.** `test_support_extraction.py` embaralha a
   fonte e exige o mesmo suporte e as mesmas coordenadas preparadas.
5. **Falha nunca vira vetor.** `test_representation_service.py` exige `FailedSupport` com a razão
   certa para suporte inencodável e saída não finita, e exige que nada seja substituído por zero ou
   default. Não existe fallback de um encoder para outro — nem no serviço, nem no composition root
   (`test_runtime_point_representation.py`).
6. **Um run incompleto não é publicado.** O writer recusa publicar quando o stream não corresponde às
   métricas reportadas, e um `add` recusado é repetível porque o estado só muda depois da validação.
7. **`__DEBUG__/` não é contratual.** Não entra no inventário, e o reader recusa explicitamente ler um
   registro de dentro dele.
8. **O manifest referencia o mapa, nunca o copia.** E o executor recusa, antes de escrever qualquer
   coisa, um run de associação construído sobre outro mapa geométrico.
9. **Identidade de código é obrigatória.** Um run sem `code_version` não existe: o estágio fica de
   fora da composição, nomeando o motivo.
10. **O estágio é opcional e desligado por padrão.** Ligá-lo não muda nenhum outro artifact do run.

## Regressões protegidas

| Falha observada | Teste |
| --- | --- |
| run publicado citando a associação de um mapa e registrando outro | `tests/runtime/test_runtime_point_representation.py` (associação sobre outro mapa recusada antes de publicar) |
| associação vazia produzindo run inconsistente | mesmo arquivo (associação vazia publica run sem representações) |
| encoder aprendido sem runtime substituído pelo determinístico | mesmo arquivo (fica de fora, nunca substituído) |
| payload truncado ou referência de payload de outro run aceitos na leitura | `test_point_representation_run_artifact.py` |
| vetor não finito persistido | mesmo arquivo (recusa de NaN/inf) |
| checkpoint divergente do hash fixado, ou com a head de classificação | `test_ptv3_pointcept_runtime.py` |

## Como rodar

```bash
pytest tests/point_representation                      # a capability
pytest tests/runtime/test_runtime_point_representation.py \
       tests/end_to_end/test_point_representation_stage.py   # composição e CLI
pytest tests/architecture                              # fronteiras
pytest -k point_representation                          # tudo que menciona a capability
```

`test_ptv3_pointcept_real.py` é skipado a menos que `CONTEXTMAP_POINTCEPT_ROOT` e
`CONTEXTMAP_PTV3_CHECKPOINT` estejam definidos, com CUDA, torch, spconv e Pointcept presentes.

O gate canônico é `make check` (ruff, ruff format, mypy, pytest).

## Mudanças estruturais

Para refatoração puramente estrutural ou documental, não se inventa um `Red` artificial: os contratos
acima já estão protegidos. O critério é o inverso — **nenhuma asserção de teste pode mudar**. Se uma
asserção ou uma fixture precisa ser ajustada para a refatoração passar, isso é prova de que a
semântica quebrou, não de que o teste estava velho.

Para mudança comportamental determinística, vale Red → Green → Refactor normal. Para mudança de
qualidade de modelo, vale a exceção de `AGENTS.md` §9: hipótese, baseline, variável alterada,
dataset, métrica e critério de aceite definidos **antes** da implementação.
