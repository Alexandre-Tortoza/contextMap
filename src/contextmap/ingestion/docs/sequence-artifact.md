# SequenceArtifact

## Por que existe

`SequenceArtifact` é o resultado persistente e reabrível de uma ingestão. Sua função principal é
desacoplar a pipeline da fonte original.

```text
ROS bag / dataset
      |
      | leitura e normalização uma vez
      v
SequenceArtifact
      |
      +-> state estimation
      +-> geometric mapping
      +-> sensor association
      +-> percepção / experimentos
```

Downstream não precisa reexecutar decoding de ROS para cada experimento.

## Layout atual

```text
<artifact>/
├── manifest.json
├── index.jsonl
├── rgb/
│   └── <observation-id>.bin
├── pointcloud/
│   └── <observation-id>.bin
├── calibration/
│   └── calibration.json
├── provenance/
│   └── provenance.json
└── diagnostics/
    ├── summary.json
    ├── warnings.jsonl
    ├── synchronization.jsonl
    ├── dropped-events.jsonl
    └── frame-graph.json
```

`manifest.json` e `index.jsonl` formam o núcleo estrutural. Diretórios e arquivos adicionais só
aparecem quando a sequência possui aquele conteúdo.

O formato detalhado e a versão de schema continuam documentados em [artifact.md](artifact.md).

## Persistência por modalidade

- imagem: metadata no índice e bytes em `rgb/`;
- LiDAR: metadata/layout no índice e bytes em `pointcloud/`;
- IMU: registro inline no índice;
- pose externa: registro inline no índice.

Essa separação evita inflar o índice com payloads grandes e mantém registros pequenos diretamente
consultáveis.

## Escrita

`SequenceArtifactWriter` é streaming:

1. cria um diretório temporário no primeiro uso;
2. grava payload e linha do índice conforme cada observação chega;
3. mantém em memória apenas metadata necessária;
4. grava calibração/provenance/diagnostics opcionais;
5. constrói o manifest e inventário;
6. verifica referências internas;
7. publica atomicamente sem substituir artifact existente.

O diretório final não deve existir parcialmente.

## Imutabilidade

Um artifact finalizado não é alterado para corrigir uma execução. Uma nova ingestão produz nova
identidade/linhagem conforme as regras do projeto.

## Leitura

`SequenceArtifactReader` oferece diferentes custos de acesso:

- `iter_index()`: percorre somente metadata do índice;
- `observation_at(offset)`: decodifica uma entrada;
- `get_observation(id)`: resolve por índice e lê somente a observação;
- `list_observations()`: materializa a sequência inteira;
- `read_calibration()`, `read_provenance()` e `read_diagnostics()`: metadata de sequência;
- `verify_integrity()`: inventário e referências internas.

Use `list_observations()` somente quando o payload completo realmente for necessário.

## Seleção e replay

`sequence_observation_selection.py` resolve:

- sequência inteira;
- faixa de posição;
- faixa de timestamp em um clock;
- conjunto explícito de observation IDs.

A decisão de quais observações pertencem à seleção usa o índice. Payload é carregado de forma lazy
quando a observação selecionada é acessada.

## Integridade

A verificação cobre:

- arquivos ausentes;
- tamanho/hash divergente;
- `payload_path` inválido;
- referência do índice para payload fora do inventário;
- caminho que tenta escapar da raiz do artifact;
- schema incompatível no momento de abertura.

Integridade não é correção: artifact inconsistente deve ser rejeitado, não reparado silenciosamente.

## Conteúdo opcional

- calibração;
- provenance;
- diagnostics;
- decisões de sincronização;
- dropped events;
- frame graph.

Esses dados enriquecem auditabilidade, mas o writer de baixo nível permite artifacts que não
possuam todos eles. O fluxo normal do runtime deve registrar o necessário para reprodução.

## O que não pertence ao artifact de ingestion

`SequenceArtifact` não é o `ContextMapArtifact` final. Ele não deve conter:

- entidades semânticas persistentes;
- mapa geométrico consolidado;
- features/claims de percepção como verdade de ingestion;
- relações espaciais;
- inferências de contexto.

Ele preserva a evidência normalizada que alimenta essas etapas.

