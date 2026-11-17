# Exemplo atual do DAG de percepção

Este modelo mantém os parâmetros e placeholders científicos documentados no [exemplo anterior](../v0.2.0/README.md), mas resolve o perfil `canonical/1` com três produtores persistentes: `region_perception`, `visual_perception` e `semantic_interpretation`. O exemplo da v0.2.0 permanece congelado como snapshot do plano anterior; este diretório não é uma release.

```bash
contextmap inspect config -c examples/perception-dag/canonical-config.toml --json
contextmap inspect plan -c examples/perception-dag/canonical-config.toml
```

[`effective-config.json`](effective-config.json) e [`resolved-plan.txt`](resolved-plan.txt) são as saídas geradas desses comandos. Os placeholders de checkpoint, revisão e limites impedem que o modelo seja executado sem parâmetros reais do run. A evidência do contrato de release v0.1.0 continua em [`examples/v0.1.0`](../v0.1.0/).
