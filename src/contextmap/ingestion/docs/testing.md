# Estratégia de testes de Ingestion

## Objetivo

Os testes de ingestion devem provar que uma fonte suportada é convertida em evidência canônica
reproduzível sem vazar detalhes do backend e que o `SequenceArtifact` preserva essa evidência com
integridade.

A inspeção visual de uma sequência é útil para debugging, mas não substitui invariantes executáveis.

## Mapa de testes

| Área | Testes principais |
| --- | --- |
| observações canônicas | `tests/ingestion/test_models.py` |
| validação | `tests/ingestion/test_validation.py` |
| timestamp | `tests/ingestion/test_timestamp_policy.py` |
| sincronização | `tests/ingestion/test_synchronization.py` |
| calibração | `tests/ingestion/test_calibration.py` |
| refinamento | `tests/ingestion/test_calibration_refinement.py` |
| artifact | `tests/ingestion/test_sequence_artifact.py` |
| seleção | `tests/ingestion/test_sequence_selection.py` |
| diagnostics | `tests/ingestion/test_diagnostics.py` |
| provenance | `tests/ingestion/test_sequence_provenance.py` |
| ROS 1 | `tests/ingestion/adapters/test_ros1_bag.py` |
| ROS 2 | `tests/ingestion/adapters/test_ros2_bag.py` |
| pose file | `tests/ingestion/adapters/test_pose_file.py` |
| integração por janela | `tests/ingestion/test_windowed_ingestion_integration.py` |
| fronteiras arquiteturais | `tests/architecture/test_boundaries.py` |

## Invariantes que os testes devem proteger

### Boundary da fonte

- nenhum tipo ROS atravessa `SourceAdapter`;
- ROS 1 e ROS 2 produzem a mesma forma canônica para modalidades equivalentes;
- mensagem malformada segue a política documentada de erro/warning;
- tópico obrigatório ausente falha antes de uma sequência incompleta ser aceita.

### Timestamp

- clocks diferentes nunca são comparados como se fossem iguais;
- correções são explícitas;
- valor bruto permanece recuperável em provenance;
- ordenação usa nanossegundos inteiros quando precisão de epoch importa.

### Calibração

- modelos de câmera não são convertidos com perda;
- quaternions/transforms inválidos são recusados;
- refinement só aplica sobre a base e alvo esperados;
- serialização round-trip preserva identidade e conteúdo relevante.

### Artifact

- round-trip preserva campos das observações;
- leitura funciona sem a fonte original;
- escrita falha não publica artifact parcial;
- artifact existente não é substituído;
- hashes e cross-references detectam corrupção;
- seleção não carrega payload não solicitado;
- caminhos persistidos não escapam da raiz.

### Sincronização

- resultado é determinístico;
- empate possui regra estável;
- ausência de candidato é explícita;
- evento de outro clock não é associado;
- diagnostics explicam decisões e drops.

## Testes com dados sintéticos e reais

Fixtures sintéticas são adequadas para invariantes exatas, edge cases e regressões pequenas.

Dados reais são úteis para validar:

- formatos efetivamente encontrados nos datasets;
- quantidade/escala;
- convenções de calibração;
- comportamento de memória e tempo;
- regressões observadas durante experimentos.

Um teste real não deve substituir o unit/contract test que explica qual regra está sendo protegida.

## Mudanças estruturais

Uma refatoração sem mudança comportamental deve manter os testes existentes verdes. Não é necessário
inventar um Red artificial para alteração puramente editorial ou reorganização que já possui
cobertura, mas paths/imports e arquitetura afetados devem ser validados.

## Checklist antes do MR

1. executar os testes focados de `tests/ingestion/`;
2. executar testes de arquitetura e packaging afetados;
3. confirmar que consumers usam a fachada pública quando apropriado;
4. confirmar que nenhum adapter vazou SDK pela API;
5. revisar docs e code map contra a árvore real;
6. executar `make check`.

O MR só deve declarar a refatoração concluída quando o gate global estiver verde.

