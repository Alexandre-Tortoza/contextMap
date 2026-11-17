# Invariantes científicos

Este documento registra o comportamento de `visual_perception` que uma reorganização de módulos
não pode alterar. Ele descreve o código materializado da branch, não uma topologia futura.

## Imagem e regiões

- `PreparedImage` preserva a observação fonte, dimensões, regiões válidas/excluídas e a cadeia
  ordenada de transformações.
- `RegionId` pertence ao produtor regional e preserva a observação física; não cria identidade persistente de entidade.
- Uma caixa não implica máscara. Grounding por caixa, proposta por ponto e região refinada seguem
  sendo evidências diferentes.
- Discovery preserva ordem de passes, tiling, política de borda, remapeamento de máscara,
  normalização, rejeições, merge e `RegionDiscoveryIdentity`.

## Features e semântica

- `FeatureScope`, forma, normalização, `EmbeddingSpace`, geometria da grade densa e referência
  de payload mantêm a mesma semântica.
- Pooling continua mask-aware e não altera a região congelada que lhe serve de suporte.
- Requests semânticos preservam identidade, views selecionadas, política de prompt, parsing,
  alternativas, abstenções, referências de resposta bruta e classificação de falhas.
- Scores de detector, similaridade, pooling e interpretação não são convertidos entre si.

## Execução e artifact

- O preset V1 permanece `canonical/1`, com os estágios e dependências atuais.
- Falha de um estágio gera `StageOutcome`; branches independentes continuam executáveis e
  dependentes são ignorados conforme o grafo atual.
- Os três artifacts de percepção mantêm contratos separados, hashes, inventory, publicação
  atômica e leitura isolada. `PerceptionResult` é somente uma view derivada. `outputs/` e `metrics/` são contratuais; `debug/` não é entrada downstream.
- O runtime continua escolhendo e compondo adapters; a capability não lê configuração externa
  nem importa o runtime.

## Verificação durante a refatoração

Cada extração deve conservar os testes existentes do subdomínio e, quando aplicável, conferir
identidades, ordenação, serialização, digest do pipeline, streams persistidos e round-trip do
artifact. Qualquer divergência é uma mudança comportamental e deve ser separada desta série.
