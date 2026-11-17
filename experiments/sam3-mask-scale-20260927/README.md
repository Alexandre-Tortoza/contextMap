# Escala de `masks_logits` do SAM3 (issue #610)

Esta investigação mede a semântica real dos tensores de máscara que o SDK do SAM3 devolve, antes de
decidir se o corte com `mask_threshold` está certo. Ela segue o modelo da #518: medir e registrar
primeiro, corrigir depois.

> **Estado: preparado, ainda não executado.** O probe e a fixture estão prontos e testados
> (`tests/evaluation/test_sam3_mask_scale_probe.py`), e o teste de contrato já existe em
> `tests/visual_perception/backends/test_sam3.py`, pulado enquanto não houver fixture real. Não
> há medição: o checkpoint do SAM3 e a GPU não estavam no ambiente em que isto foi preparado.
> Nenhuma conclusão sobre a escala está registrada ainda.

## Problema

`Sam3ImageProcessorRuntime` (`src/contextmap/visual_perception/backends/sam3.py`) lê o tensor de
máscara com `output.get("masks_logits", output.get("masks"))` e binariza cada pixel com
`x >= mask_threshold`, sem sigmoid e sem detecção de escala. `mask_threshold` tem default `0.5`,
é validado em `[0, 1]` e entra na metadata da proposta e no digest de `Sam3Config`. A
documentação (`src/contextmap/visual_perception/docs/region-discovery.md`) chama os valores de
"probabilidades de máscara". O único teste desse caminho usa um fake entre 0,1 e 0,8.

- Se `masks_logits` já são **probabilidades** (pós-sigmoid), o código está certo.
- Se são **logits** (pré-sigmoid, sem limite), o default 0,5 corta em `sigmoid(0.5) ≈ 0,62`, não
  em 0,5. Com a validação em `[0, 1]`, só se expressam cortes entre `sigmoid(0) = 0,5` e
  `sigmoid(1) ≈ 0,73`.

## O que o probe mede

`scripts/sam3_mask_scale.py` roda o processor oficial do SAM3 **pelo caminho do próprio runtime**.
O processor é envolvido por um proxy que só grava a saída bruta de `set_text_prompt` e é passado a
`Sam3ImageProcessorRuntime.predict`. A chamada percorre:

- a verificação de placement do modelo;
- `torch.inference_mode`;
- o autocast `bfloat16`;
- `set_image`, `set_confidence_threshold` e `set_text_prompt`;
- a binarização atual.

`sam3.py` não muda.

Cada chamada usa um `Sam3Config` de uma só frase (`prompts=(prompt,)`). Desde a #662 o runtime
consulta um conjunto de frases sobre a mesma imagem codificada, mas o probe mantém uma chamada por
frase, para que cada saída bruta pertença a um único par frame × prompt.

Para cada chamada (frame × prompt), e somado sobre todas:

- para `masks_logits` e para `masks`: dtype, shape, mínimo, máximo, histograma com bins fixos
  (bordas em −20, −10, −5, −2, −1, 0, 0,25, 0,5, 0,75, 1, 2, 5, 10, 20; os extremos são
  abertos), valores abaixo de 0 e acima de 1, fração fora de `[0, 1]` e se o tensor é binário;
- a binarização pela regra atual (`x >= mask_threshold`) e pela regra sigmoid
  (`sigmoid(x) >= mask_threshold`): primeiro plano de cada uma e pixels em que discordam. Isso
  quantifica o impacto do corte deslocado, como pede o plano de validação da issue;
- quando o SDK devolve `masks` binárias: a fração de pixels em que cada regra concorda com elas;
- o primeiro plano que o runtime atual realmente produziu, que tem de ser igual ao da regra atual.

**Regra de escala** (gravada nos dois documentos): `probability` quando todo valor finito do
tensor que o runtime consome está em `[0, 1]`; `logit` quando algum está fora; `undetermined`
sem valor finito. O runtime consome `masks_logits` quando existe, senão `masks`.

**Identidade registrada**:

- checkpoint: revisão `refs/main` do cache do Hugging Face e o blob (SHA-256, para arquivo LFS) de
  cada arquivo do snapshot, lido sem abrir os pesos;
- SDK: versão instalada do `sam3` e commit Git, quando instalado de um checkout;
- `contextmap`: versão e commit;
- torch: versão, CUDA e GPU;
- `nvidia-smi`: GPU, driver e memória;
- a sequência de origem e o SHA-256 de cada imagem.

## Conjunto declarado

| | |
|---|---|
| Sequência | `SequenceArtifact` `720a486de8d44c16a9d3d2ff9fa7b1a4` (corridor-02) |
| Frames | `camera_1_image_raw-000215`, `camera_1_image_raw-001295`, `camera_1_image_raw-002267`: o 1º, o 11º e o 20º dos 20 frames congelados no cenário 1.0.4 (um teste confere) |
| Prompts | `floor`, `wall`, `door`: superfícies que todo frame do corredor mostra. Não foram ajustados: só precisam produzir máscaras |
| Configuração | `Sam3Config` com `checkpoint=facebook/sam3`, `precision=bfloat16`, `device=cuda`, `strategy=text_prompt`, `score_threshold=0.5` (limita às propostas confiantes; a escala da máscara não depende dele), `mask_threshold=0.5` (o default) |

Pelo registro do #338 em `region-discovery.md`, o modelo de imagem oficial só executa sob autocast
`bfloat16`; em `float32` o SDK falha. Tudo isso é argumento do comando e fica gravado no resultado.

## Como executar (máquina com o checkpoint)

**Pré-requisitos**

- `contextmap` com o extra `vision`, torch com CUDA, Pillow e o SDK do SAM3
  (`facebookresearch/sam3`).
- Acesso ao repositório *gated* `facebook/sam3` no Hugging Face, com o checkpoint no cache local.
- A revisão do checkpoint, que entra como `--model-version`, porque `Sam3Config` não aceita versão
  implícita:
  `cat ~/.cache/huggingface/hub/models--facebook--sam3/refs/main`.
- O `SequenceArtifact` acima em `outputs/`.

**Comando**, da raiz do checkout principal:

```bash
python experiments/sam3-mask-scale-20260927/scripts/sam3_mask_scale.py \
    --sequence outputs/ingest-real/sequences/corridor-02/720a486de8d44c16a9d3d2ff9fa7b1a4 \
    --model-version <revisão do checkpoint>
```

Por padrão, o processor é construído como o README do SDK documenta:
`Sam3Processor(build_sam3_image_model())`. Se o carregamento local for outro (outro checkpoint,
outro device), use `--processor módulo:atributo`, um callable sem argumentos que devolve o
processor oficial já carregado. As demais opções são `--frames`, `--prompts`, `--device`,
`--precision`, `--score-threshold`, `--samples` (padrão 3), `--window` (padrão 16 pixels),
`--run-number` e `--outputs-root`.

Saídas, em `outputs/sam3-mask-scale-20260927/run-0001/`. Um diretório existente é recusado, e
nada é sobrescrito:

- `measurement.json`: tudo o que foi medido;
- `samples.json`: a fixture. É escrita só quando a escala é determinada e alguma proposta tem
  máscara com borda; sem ela, o comando sai com 1.

## O que volta para o repositório

1. `measurement.json` → `experiments/sam3-mask-scale-20260927/results/measurement.json`.
2. `samples.json` → `tests/fixtures/sam3_mask_scale/samples.json`. Com a fixture presente, o teste
   `test_the_runtime_thresholds_masks_logits_on_the_scale_measured_from_the_real_sdk` deixa de ser
   pulado:
   - com **probabilidades**, passa de imediato e documenta a interpretação. A falta de Red se
     justifica pela natureza de investigação, como a issue prevê;
   - com **logits**, falha contra a comparação direta atual. É o Red da issue de correção. Os
     testes `test_the_contract_test_*` já mostram, com dados sintéticos, que ele discrimina as
     duas escalas.
3. Uma seção "Resultado" neste README com a escala medida, os números e a decisão de correção, e a
   correção de `region-discovery.md` se "probabilidades" não corresponder ao tensor consumido.

### Formato da fixture

`samples.json` guarda:

- `observed_scale`, a regra, `checkpoint`, `model_version`, `device`, `precision`,
  `score_threshold` e `mask_threshold`;
- `config_digests` por prompt e a identidade;
- `samples`, até `--samples` janelas de no máximo `--window` × `--window` pixels.

Cada amostra é a proposta de maior score de uma chamada e traz:

- `frame`, `prompt`, `proposal_index`, `box`, `score`;
- `window` (`[x_min, y_min, x_max, y_max]`, semiaberta), `width` e `height`;
- `masks_logits` (valores exatos, como float64 do valor do SDK) e o dtype original;
- `masks`, quando alinhadas, e o dtype delas;
- `full_min` e `full_max` do tensor inteiro da proposta.

Sob escala `logit`, a janela é centrada num pixel em que as duas leituras divergem
(`0 <= x < mask_threshold`), para que o teste fixe exatamente o que uma correção mudaria. Sob
`probability`, a janela é centrada na borda da máscara.

## Decisão de correção (da issue; depende da medição)

- **Probabilidades confirmadas**: alinhar nomes, testes e documentação, e manter a comparação
  atual.
- **Logits confirmados**: aplicar sigmoid antes do corte, ou consumir só `masks` (se a medição
  as confirmar binárias), ou nomear a escala explicitamente na configuração. A decisão fica
  versionada no digest de `Sam3Config`, com issue de implementação própria.

## Fora de escopo (da issue)

- Implementar a correção decidida: terá issue própria depois da medição.
- Mudar a representação de máscara (`InlineMask`) ou a normalização de regiões.
- A semântica de threshold dos demais backends (SAM2, Florence-2).
- Tuning científico do valor do threshold.

## Scripts e testes

| Arquivo | Faz |
|---|---|
| `scripts/sam3_mask_scale.py` | o probe: carrega o processor, roda o caminho do runtime, mede, escolhe as janelas e grava a medição e a fixture |
| `tests/evaluation/test_sam3_mask_scale_probe.py` | estatísticas, regra de escala, comparação das binarizações, janelas, formato e exatidão da fixture, identidade do checkpoint e o probe inteiro com um processor falso |
| `tests/visual_perception/backends/test_sam3.py` | o teste de contrato sobre a fixture real (pulado sem ela) e os dois testes que mostram que ele discrimina as escalas |
