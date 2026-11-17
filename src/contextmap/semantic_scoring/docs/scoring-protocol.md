# Pontuação semântica: estágio opcional sobre produtores pinados (#652)

Os scorers CLIP e AlphaCLIP (#113) dão a uma hipótese semântica um canal de evidência independente do VLM: a similaridade de cosseno entre o texto da hipótese e a feature visual da região que ela descreve. Esta página registra onde eles rodam, o que pontuam, de onde vem o texto e o que o run persiste. Ela é a nota de design das decisões D1, D2 e D3 da #652.

## Decisões

### D1: um estágio próprio, sobre uma seleção explícita de produtores pinados

O scorer roda no estágio opcional `semantic_scoring` do perfil `canonical/1`, desligado por padrão, e não dentro do grafo de estágios de `visual_perception`. O estágio consome `RegionPerceptionRunArtifact`, `VisualEvidenceArtifact` e `SemanticInterpretationArtifact` explicitamente selecionados e publica um `SemanticScoringRunArtifact`.

- A configuração do scorer é de outro estágio, então nunca entra no `config_digest` de `visual_perception`. Dois braços que só diferem no scorer reusam, por conteúdo, os mesmos produtores regional, visual e semântico.
- O VLM roda uma vez. Com um VLM não determinístico (#580: 29/90 = 32,2% de concordância exata entre dois runs idênticos), recomputar a percepção em cada braço confundiria o efeito do scorer com a reamostragem das claims.
- Desligar o scorer preserva os produtores selecionados: a configuração sem o estágio resolve exatamente como antes, e o exemplo congelado da v0.1.0 só ganha `pipeline.stages.semantic_scoring = false`.
- Na expansão de fases (#527), ligar o scorer insere um estágio. "Scoring off × scorer" é, portanto, um fator `topology` de uma fase de composição, e o #545 exige que os três produtores a montante sejam pinados.

O custo é um contrato de artifact novo e um estágio no catálogo e na composition root. Semantic Fusion pode consumir esse artifact como canal opcional sob sua política versionada, preservando a escala e a provenance do score. O contrato `SemanticSupport` não é reaproveitado por coerção.

### D2: só claims de região, contra a feature da mesma região final

Cada claim de região (primária ou alternativa) é pontuada contra a feature de região **da mesma região**, já extraída pelo slot `visual_features.region` no produtor visual pinado. O scorer não codifica imagem.

- CLIP pontua a feature do crop da caixa (`ClipRegionSemanticScorer`); a máscara é ignorada, então o fundo dentro da caixa entra no score. AlphaCLIP pontua a feature condicionada à máscara (`AlphaClipSemanticScorer`).
- Pontuação de cena contra uma feature CLIP global fica N/A: nenhum slot do runtime produz a feature global (o slot de região fixa o escopo `region`). `ClipSemanticScorer`, o scorer global, continua existindo para o grafo interno, mas nenhum componente do runtime o seleciona. As claims de cena ficam sem score e são contadas à parte.
- Uma região sem feature compatível deixa as suas claims sem score. A ausência é o registro; nenhum zero é fabricado.
- Os espaços CLIP e AlphaCLIP nunca se misturam: uma feature de região em outro espaço de embedding é recusada no primeiro request com `SemanticScoringError`, e o estágio falha sem publicar nada.
- Cada execução visual possui seu extrator de região selecionado. Braços CLIP e AlphaCLIP podem selecionar artifacts visuais distintos sobre o mesmo produtor regional e reutilizar exatamente o mesmo produtor semântico, mantendo as claims pinadas. Os respectivos embedding spaces e scorers precisam ser compatíveis; não há coerção entre espaços.

### D3: uma torre de texto por família, no mesmo espaço da feature

| Família | Adapter de texto | Runtime | Espaço declarado |
|---|---|---|---|
| CLIP | `ClipTextEncoder` (`adapters/clip_text.py`) | `HuggingFaceClipTextRuntime`: carrega só `CLIPTextModelWithProjection` e o tokenizer, no checkpoint e na revisão fixados, sem download implícito; ou um provider (`ClipTextRuntime`) | `clip_embedding_space(checkpoint, revision, dimensão, "l2")`, a mesma identidade das features de imagem do mesmo checkpoint e revisão |
| AlphaCLIP | `AlphaClipTextEncoder` (`adapters/alphaclip_text.py`) | provider (`AlphaClipTextRuntime`), por exemplo `OfficialAlphaClipTextRuntime(config=..., checkpoint_root=...)`, que verifica o fingerprint combinado base+alpha e usa a torre de texto base do pacote oficial | `alphaclip_embedding_space(model_name, visual_code_version, checkpoint_fingerprint, dimensão, "l2")`, a mesma identidade das features do extrator de região |

As duas funções de espaço são as mesmas que os extratores de imagem usam, então texto e imagem só se comparam quando todo campo coincide. A AlphaCLIP muda só o encoder de imagem: a projeção condicionada ao alpha foi treinada contra a torre de texto do checkpoint base, e por isso o texto usa essa torre e declara o espaço AlphaCLIP. O runtime da AlphaCLIP vem do chamador porque os checkpoints locais vivem numa raiz que a configuração não declara, a mesma situação da feature AlphaCLIP (#71).

- **Template do texto.** `text_template` é obrigatório e contém `{hypothesis}` exatamente uma vez, sem conversão nem especificação de formato (`"{hypothesis}"`, `"a photo of a {hypothesis}"`). Ele muda todo valor, então faz parte da identidade do scorer e nunca tem default.
- **Truncamento.** Um texto maior que o contexto da torre é recusado (`ClipTextInferenceError`/`AlphaClipTextInferenceError`), nunca truncado em silêncio.
- **Normalização.** O texto é normalizado em L2 em `float32`, qualquer que seja a precisão da inferência, porque o scorer exige norma 1 com tolerância de `1e-5`.
- **Identidade.** `clip_region_scorer_provenance(config)` e `alphaclip_region_scorer_provenance(config)` dão a `BackendProvenance` (`capability="semantic_scorer"`) conhecida antes de qualquer modelo carregar: o modelo, a revisão (CLIP) ou o fingerprint dos pesos (AlphaCLIP) e o fingerprint de toda configuração que muda um valor (template, precisão, device, lote, versão do adapter de texto e regra de pontuação `cosine_similarity:region/1`). O executor usa essa proveniência no run id e na chave de reuso (#636).

## Configuração

```toml
[pipeline.stages]
semantic_scoring = true

[components.semantic_scoring.scorer]
backend = "clip"

[components.semantic_scoring.scorer.clip]
checkpoint = "openai/clip-vit-large-patch14"   # o mesmo das features de região
revision = "<sha de 40 caracteres>"            # a mesma revisão das features de região
text_template = "a photo of a {hypothesis}"
```

Para AlphaCLIP, o bloco `alphaclip` declara `model_name`, `base_checkpoint_path`, `alpha_checkpoint_path`, `checkpoint_fingerprint`, `text_template` e, se o extrator não usar o default, `visual_code_version`; o runtime vem de `resources.providers` (ou `providers=` num chamador Python).

## `SemanticScoringRunArtifact` 0.2.0

```text
<run>/semantic_scoring/
├── manifest.json
├── README.md
└── outputs/
    └── scores.jsonl
```

- `outputs/scores.jsonl`: um `SemanticScore` canônico por linha (score, claim, feature, tipo, valor, `calibrated_probability` sempre `null`, espaço, observação, região e proveniência do scorer), na ordem das observações e das claims. Claims e features permanecem em seus respectivos artifacts e são referenciadas por identidade.
- `manifest.json`: `lineage` (`sequence_artifact_id`, `selection_id`, `region_artifact_id`, `visual_evidence_artifact_id`, `semantic_interpretation_artifact_id`), `scorer`, `embedding_space` completo e `embedding_space_id`, `text_template`, `scope` (`region`), `score_type` e `score_semantics` (cosseno bruto em `[-1, 1]`, não uma probabilidade, não calibrado, nunca comparável entre espaços), `counts` (resultados; claims de região, de cena, pontuadas e não pontuadas; scores), `code_version`, `schema_version`, `created_at` e o inventário com tamanho e SHA-256. Não há `debug/`.

`SemanticScoringRunWriter` publica pelo `AtomicRunDirectory` e nunca substitui um run existente. Ele recusa uma view de outra seleção de produtores ou uma observação repetida, e todo score que o run não pode garantir: de outro scorer ou espaço, com probabilidade calibrada, que liga uma claim à feature de outra região ou que referencia o que o resultado não tem (a regra de `PerceptionResult` é aplicada como está). `SemanticScoringRunReader` recusa uma versão de schema desconhecida e lê os scores linha a linha.

## Execução

`SemanticScoringExecutor` abre os três produtores e valida sequência, seleção e artifact
regional de origem. Constrói uma view derivada com as regiões reais, features e claims,
compõe o scorer sobre o `feature_store()` visual e entrega a view e os scores ao writer.
Nenhum upstream é reescrito. Veja [composição do runtime](../../runtime/docs/executors.md).

## Fora de escopo

- avaliação do scoring contra o reference set (ranking entre alternativas anotadas, AUROC/AUPRC), que é da Evaluation (#658): ver [`evaluation/docs/semantic-scoring.md`](../../evaluation/docs/semantic-scoring.md);
- calibração do cosseno em probabilidade (`evaluation/score_calibration.py`);
- SigLIP2 como scorer.
