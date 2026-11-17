# Mapa de código — Region Perception

| Subsistema | Responsabilidade |
|---|---|
| `region_evidence.py`, `region_geometry.py` | Contratos de regiões, candidatos, máscaras e provenance |
| `evidence_identity.py` | IDs por run regional/observação/origem e remapeamento no boundary |
| `region_assembly.py` | Ordenação das fontes regionais sob `region-set-assembly/1` |
| `discovery/` | Planejamento de passes, remapeamento, normalização, rejeições e merge auditável |
| `grounding/` | Query explícita, inputs content-addressed, outputs caixas/pontos e codecs |
| `refinement/` | Prompts, regra de aceitação de máscaras e outcomes com scores nativos |
| `region_semantic_hint.py` | Texto nativo de proposta como evidência, sem criar claims |
| `region_eligibility.py` | Motivo explícito de inelegibilidade de uma região para um consumidor |
| `ports.py` | Contratos substituíveis consumidos por composição runtime |
| `adapters/` | SAM2, SAM3, Florence2, LocateAnything e materialização de imagem |
| `artifacts/region_serialization.py` | Codec regional e tradução do audit local para lineage publicado |
| `artifacts/frame.py` | Outcome por observação e metadados de preparo |
| `artifacts/run_manifest.py` | Identidade, seleção, configuração, schema e inventário |
| `artifacts/run_writer.py`, `run_reader.py` | Publicação atômica e acesso tipado |
| `artifacts/mask_store.py` | Payloads `.npy`, índice por observação/identidade e hashes |

## Layout publicado

```text
manifest.json
outputs/
    frames.jsonl
    regions.jsonl
    region-discovery-audit.jsonl
    grounding.jsonl
    grounding-responses/response-NNNNNN.txt
    refinement.jsonl
    masks/
        mask-index.jsonl
        <observação>/<identidade>.npy
```

Os streams existem mesmo sem execuções opcionais. O store de máscaras existe quando há
payloads. `debug/` não é entrada downstream. Nenhum SDK, tensor nativo ou estado de backend
atravessa o contrato de artifact.
