# relation-graph-diagnostic-20260929

Diagnóstico offline, read-only, do grafo de relações do run
`outputs/canonical-2-real-20260929/`. Separa as hipóteses concorrentes para o
tamanho do grafo de cena (fragmentação de entidades, densidade de candidatos,
multiplicador estrutural, materialização) **sem alterar código de produção nem os
artefatos do run**. Não inclui calibração de associação por aparência (H5).

## Conteúdo

- `scripts/diagnose_relation_graph.py` — análise; recebe `<run_workspace>` e
  `<out_json>`, lê apenas os `outputs/*.jsonl` públicos e o context map (via
  `ContextMapArtifactReader`), escreve o relatório só neste diretório.
- `report.json` — métricas legíveis por máquina (H1, normalização, H2, H3, H4 e a
  classificação estrutural por entidade).
- `REPORT.md` — leitura humana, conclusões e ressalvas.

## Executar

```bash
PYTHONPATH=src python3 \
  experiments/relation-graph-diagnostic-20260929/scripts/diagnose_relation_graph.py \
  outputs/canonical-2-real-20260929 \
  experiments/relation-graph-diagnostic-20260929/report.json
```

Requer Python 3.11+ e `PYTHONPATH=src` (usa `contextmap.artifact`). Não usa GPU
nem modelos. Para o run conhecido, um desvio nos valores esperados (nº de
entidades, singletons, relações canônicas e `by_state`) aborta com erro — sinal
de carregamento/canonicalização incorretos, não de dado válido.

## Entradas lidas (somente leitura)

- `pipeline/entity_resolution/outputs/resolved-entities.jsonl`
- `pipeline/spatial_relations/outputs/relations.jsonl`
- `pipeline/context_map/` (via `ContextMapArtifactReader`)

## Ressalva de método

A classificação estrutural (H3) é por palavra-chave sobre rótulos de texto livre
e é reportada como **sensibilidade** (variantes strict/loose), não como verdade
de campo — não existe tipo de superfície estrutural no domínio.
