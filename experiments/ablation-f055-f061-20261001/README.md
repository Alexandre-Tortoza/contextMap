# Ablação f055/f061: qualidade × custo da descoberta, da interpretação e da associação por caixa

Data: 01/10/2026. Tudo local, sem commit. Identidade de código em `code-identity.txt`
(`4b67748+local.51f44dbb81b3`). Comandos exatos em [`commands.sh`](commands.sh); ambiente em
[`env.sh`](env.sh).

## Objetivo

Comparar, com o **mesmo par de frames** e mapas finais voxelizados, combinações das capacidades que já
existem: descoberta (SAM2, Florence-2, SAM3), interpretação (Qwen3-VL-4B, Florence-2) e associação
RGB↔LiDAR (máscara ou caixa). Cada braço responde a uma hipótese declarada antes de rodar. O critério
é o mapa útil por custo de máquina, nunca "menos regiões" sozinho.

## Par de frames: N = 55, N+6 = 61 (corridor-02, 10:31)

Escolhido uma vez. f055 tem:
- duas portas azuis próximas, com barras antipânico, visores e fechadores;
- viga e luminárias no teto;
- corredor profundo (~1–20 m), entulho e oclusões parciais;
- boa luz.

Ele já tinha baseline medido (run-0006/0007), a grade SAM2 do C4 e template de anotação.
f061 = f055 + 6 (~0,25 s depois), com a câmera mais à frente: a porta esquerda passa a ocupar
x≈45–235. Todos os mapas usam a fundação voxel de 0,10 m `run-0003` (393.054 pontos) do run store
`workspace/run-frame-10_31`.

**Limite do par: nenhuma pessoa aparece.** O Human Search só mede falsos positivos aqui.

## O que foi implementado ou corrigido (código de produção, TDD)

- **`box-depth-component-v1`** (`sensor_association/membership.py`, componente opcional
  `sensor_association.membership`). Uma região só com caixa deixa de ser sempre `NO_INLINE_MASK`.
  - Os pontos visíveis da caixa são agrupados em componentes 3D: células de 0,15 m, vizinhança 26,
    no frame da câmera (`FrameProjection.camera_points_m`).
  - O suporte é o componente com mais pontos no núcleo da caixa.
  - Caixa ambígua ou sem suporte suficiente é pulada com o motivo (`AMBIGUOUS_BOX_SUPPORT`,
    `INSUFFICIENT_BOX_SUPPORT`), nunca aceita inteira. É a resposta à objeção da #649.
  - Regiões com máscara seguem `mask-membership-v1` no mesmo run.
  - Testes: `tests/sensor_association/test_box_region_membership.py` (15) e
    `tests/runtime/test_runtime_box_membership.py` (4); Red confirmado antes.
- **Nomes honestos da qualidade** (`observation-quality-v3`, `region-support-metrics-v2`):
  - `mask_area_px` → `footprint_area_px`;
  - `support_density_per_mask_pixel` → `support_density_per_footprint_pixel`;
  - mais `footprint_kind` (`mask`/`box`) e `excluded_visible_count`.
- **Matriz de capacidades**: `contextmap2.box_component_association`, com Florence-2 `<OD>` e
  LocateAnything (categoria/frase) como produtores suportados.
- **Docs**: `sensor_association/docs/{membership,quality,contracts}.md`,
  `runtime/docs/composition.md`, `evaluation/docs/capability-matrix.md`, `CHANGELOG.md`.
- **Fora do repositório**: `load_florence2` em
  `~/.cache/contextmap2-ablation/providers/ablation_providers.py`, com o snapshot
  `florence-community/Florence-2-large@4271c66b…` fixado.
- Ruff, format e mypy limpos nos arquivos tocados. A suíte inteira só falha em testes que já falhavam
  (ver "Problemas").

## Braços

| map_id (`corridor02-f055-f061-…-vox10`) | Hipótese | Variável vs. referência |
|---|---|---|
| `sam2-qwen-baseline` (B0) | referência | — (percepção da run-0006/0007 reusada; associação refeita com o código novo) |
| `sam2-qwen-minarea1536` (A1) | o piso de 0,5 % do frame corta chamadas de VLM e labels absurdas sem perder recall | `normalization_config.minimum_area_pixels = 1536` |
| `sam2-qwentight` (S1) | o contorno magenta da view piora as labels; o recorte justo melhora | view `mask_outline_crop` → `tight_crop` |
| `florenceod-florencesem-boxdepth` (F1-od) | Florence object-centric substitui o SAM2 com menos regiões e custo muito menor | sistema: Florence `<OD>` + Florence semântico + regra por caixa |
| `florencerp-florencesem-boxdepth` (F1-rp) | idem, com propostas `<REGION_PROPOSAL>` (o `<OD>` só achou 3 caixas no probe) | task `<OD>` → `<REGION_PROPOSAL>` |
| `florencerp-qwen-boxdepth` (F1q) | o Qwen é necessário, ou o Florence semântico basta | intérprete Florence → Qwen v3 (`tight_crop`), mesmas caixas |
| `sam2-qwen-baseline-relinside` (R1) | "contido_em" (`inside`) sai das entidades atuais | predicados `+ inside, on_top_of, above` |
| Human Search (sem mapa) | o canal SAM3 acha evidência direta de pessoa sem falso positivo | SAM3 `text_prompt` (person, human body, face, hand + backpack, helmet, shoe, glove) |

Não rodaram, com o motivo:
- **LocateAnything**: o grounding só se *soma* à descoberta, sem dedup, então só aumentaria o custo
  do Qwen. O 3B junto do Qwen não cabe nos 8 GB de forma estável.
- **Embeddings contra false merge**: a associação por aparência segue desligada até a calibração H5.
- **Qwen-fallback real**: o `<OD>` não devolve score, e o Florence semântico errou 6 de 6 labels.
  Não há sinal para o gate.

## Resultados (os dois frames somados)

A anotação é um **rascunho de modelo, pendente de revisão humana** (`annotations/frame-55.json`,
`frame-61.json`): 16 + 13 instâncias de objeto, mais estrutura. `recall` = instâncias de objeto com
uma região de IoU de caixa ≥ 0,5; `c4` = só door/door_handle/window; `labels` = label primária
correta entre as regiões casadas; `frag` = regiões ≥ 50 % dentro da caixa de um objeto, média por
objeto atingido. Tabela completa em `results/metrics.json`; esta tabela em `results/summary-table.md`.

| braço | regiões | < 0,5 % | frag | recall | c4 | labels | abstenções | obs. c/ suporte | puladas | entidades | nos 2 frames | false merge | false split | relações | chamadas VLM | percepção (s) | pico GPU (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 129 | 86 | 3,13 | 22/29 | 15/18 | 7/22 | 11 | 112 | 0 | 51 | 11 | 2 | 2 | 3.526 | 129 | 346,5 | ≈6.070¹ |
| S1 | 129 | 86 | 3,13 | 22/29 | 15/18 | 6/22 | 59 | 112 | 0 | 51 | 11 | 2 | 2 | 3.526 | 129 | 292,3 | 6.068 |
| A1 | 42 | 0 | 2,25 | 12/29 | 9/18 | 5/12 | 0 | 42 | 0 | 28 | 5 | 2 | 1 | 1.324 | 42 | 137,7 | 6.107 |
| F1-od | 4 | 2 | 1,0 | 4/29 | 4/18 | 0/4 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 4 | 9,4 | 5.917 |
| F1-rp | 9 | 4 | 1,0 | 6/29 | 6/18 | 0/6 | 0 | 5 | 4 | 4 | 1 | 0 | 0 | 4 | 9 | 11,0 | 5.917 |
| F1q | 9 | 4 | 1,0 | 6/29 | 6/18 | 4/6 | 0 | 5 | 4 | 4 | 1 | 0 | 0 | 6 | 9 | 41,6 | 7.314 |
| R1 | 129 | 86 | 3,13 | 22/29 | 15/18 | 7/22 | 11 | 112 | 0 | 51 | 11 | 2 | 2 | 4.813 | 129 | (B0) | — |

¹ A percepção do B0 foi reusada e não passou pelo amostrador de GPU. O S1 roda a mesma pilha e mediu
6.068 MiB; o diagnóstico do Qwen no B0 registra 5,67 GB.

Custo por estágio no B0 (frame 55): SAM2 4,3 s, features 4,9 s, Qwen 154,4 s (61 chamadas,
mediana 2,5 s). **O VLM é 94 % do tempo.** A associação com o código novo leva ~2,4 s por frame
(processo inteiro), e o build ~15 s.

Human Search:
- 0 propostas com score ≥ 0,5 nas 8 frases, nos dois frames (~15 s e ~6 GB por frame).
- Controle positivo com "door" e "window" no f055: 9 propostas → 6 regiões, as duas portas inteiras,
  a porta do fundo e as janelas. O canal detecta quando há o que detectar.

## Decisões sustentadas pelas medições (com a anotação provisória)

1. **O SAM2 super-segmenta, e o gargalo de qualidade está na interpretação, não na descoberta.**
   - Super-segmentação: ~3 regiões por objeto, até 14 na porta esquerda do f061, e 86 de 129 regiões
     abaixo de 0,5 % do frame.
   - Mesmo assim, o B0 tem o melhor recall (22/29; 15/18 no vocabulário do C4).
   - As labels é que falham: 7 de 22 corretas ("towel rack" na barra, "knife" no visor, "barrel" na
     ripa).
2. **O piso de área não serve como corte de custo.** −60 % de tempo e de chamadas, mas o recall cai de
   22 para 12 de 29, com barras, visores e luminárias perdidos. Rejeitado pelo critério declarado.
3. **Florence-2 não substitui o SAM2 neste par.**
   - É ~30× mais barato (9–11 s contra 346 s), mas acha 4–6 de 29 objetos.
   - O `<OD>` vê só janelas e a roda do robô; o `<REGION_PROPOSAL>` dá 5 caixas por frame.
4. **O Florence-2 semântico não basta como intérprete.** Sobre as mesmas caixas, acertou 0 de 6 labels
   ("refrigerator" na porta, "house"/"curtain" nas janelas); o Qwen acertou 4 de 6. Um fallback gated
   pelo Florence chamaria o Qwen quase sempre. Fraco: n = 6.
5. **O recorte justo não melhora as labels.** S1 dá 6 de 22 contra 7 de 22, e as abstenções vão de 11
   para 59: sem contexto, o Qwen abstém em vez de acertar. O contorno magenta não é a causa principal
   do erro de label.
6. **A regra por caixa funciona e é auditável.**
   - A porta direita recebe 3.396 pontos e exclui 14 pontos visíveis da parede.
   - As duas janelas de vidro do fundo são puladas como `insufficient_box_support`, coerente com o
     LiDAR atravessando vidro. Hipótese, não medida.
   - Na porta direita do f055, o suporte por caixa tem profundidade de 2,02–2,95 m; a máscara SAM2
     do B0, 0,60–2,94 m. Aqui a caixa não vazou mais que a máscara.
7. **"Contido_em" já sai do código existente** (R1). As 11 relações `inside` suportadas incluem visor
   e barra dentro da porta direita, e fragmentos da porta esquerda dentro dela, com labels erradas.
   `inside` liga os fragmentos do SAM2 ao objeto-pai. Isso é sinal de parte-todo, mas `part_of` não
   existe como predicado.
8. **Human Search não deu falso positivo** e mostrou que detecta quando há objeto. O caso positivo
   (pessoa) fica sem teste neste par.

## Inconclusivo

- **Toda métrica de recall/label/merge usa um rascunho de anotação de modelo.** Revisar
  `annotations/frame-55.json` e `frame-61.json` (sobrepostos em `*-draft-overlay.png`) e rerodar
  `scripts/arm_metrics.py`.
- Se a label melhora com mais contexto (`contextual_crop`) ou com outro VLM (Eagle 2.5, Qwen 8B): não
  testado.
- SAM3 com um vocabulário genérico de objetos como descoberta (no controle, achou as portas inteiras,
  coisa que o SAM2 fragmenta): promissor, mas fecha o vocabulário. Precisa de um braço próprio, e o
  slot de descoberta é único.
- Os 2 false merges do B0 envolvem instâncias estruturais com caixa grosseira (viga+teto,
  parede+janela). Pode ser artefato da anotação.
- Se embeddings reduzem false merge: depende da calibração H5.
- Positivo do Human Search e uma hipótese `possible_human_presence` downstream: sem pessoa no par e
  sem consumidor; nada foi criado (YAGNI).

## Viewer (validação final)

Viewer em `~/Projects/contextmap2-viewer`, `.env.ablation-f055-f061`, porta 3001. Inspeção scriptada
por CDP: `scripts/viewer_check.mjs` com o Chromium do sistema. Capturas em `results/viewer/`;
relatórios em `results/viewer/viewer-check*.json`.

Os 7 mapas aparecem no catálogo, validados e com hashes conferidos. Em B0, A1, S1 e F1q, todos os
passos passaram: escolher o mapa, focar a geometria, modo "Labels semânticas", camada de relações,
selecionar uma entidade "door" (o viewer a isola e a foca em 3D), abrir a evidência da porta direita
do f055 e focar o suporte da região. A evidência traz:
- a imagem de origem (fisheye), o timestamp e as coordenadas da região;
- o claim (hipótese, descrição, atributos);
- a qualidade (`footprint_kind`, profundidade);
- a proveniência completa: código, fingerprint, pose, calibração, `membership_policy_id`.

Problemas vistos na inspeção:
- **B0, B0-relinside e S1: o grafo de relações é ilegível**: 3.526–4.813 relações viram um novelo
  sobre a nuvem. É o achado de densidade de entidades já conhecido.
- Com o enquadramento "focar geometria" (o corredor inteiro), as cores semânticas mal aparecem: os
  pontos rotulados são uma fração pequena de 393 mil. Só focando a entidade se vê o objeto.
- A lista de evidências pagina de 50 em 50 e começa pelo f061: a porta do f055 fica na 3ª página.
- **F1-rp/F1q: a porta do fundo vista nos dois frames virou duas entidades**, e o mapa afirma
  `touching` e `within_interaction_range` entre elas (false split que a anotação não pegou).

Mapas para comparar no Viewer:
- `sam2-qwen-baseline` (referência; melhor recall);
- `florencerp-qwen-boxdepth` (mais barato com labels razoáveis; suporte por caixa);
- `sam2-qwen-baseline-relinside` (relação "contido_em");
- `sam2-qwen-minarea1536` (o que o piso de área perde).

## Problemas fora desta ablação (registrados, não corrigidos)

- Suíte inteira (`pytest tests`): **8.402 passaram, 13 falharam, 14 pulados.**
  - 12 falhas: testes de harness/end-to-end que esperam `context_map/` dentro do run de build
    (`IncompleteContextMapArtifactError: 'context_map' is not a directory`), desde a mudança local do
    maps root.
  - 1 falha: `test_no_example_file_is_ignored`, porque `examples/v0.2.0/corridor-02-image-masks.toml`
    está ignorado pelo git.
  - As duas famílias vêm de trabalho local anterior; nenhuma toca membership nem qualidade.
  - `make check` não roda neste ambiente (sem venv do projeto). Rodaram ruff + format na árvore
    inteira e mypy nos pacotes tocados, todos limpos.
- Uma camada de config que troca `view_policy.region_views` herda `mask_outline` da camada de baixo e
  é recusada (merge chave a chave). Os braços usam `--set …mask_outline=null`.

## Arquivos

- `config/`: camadas por braço.
- `annotations/`: frames, rascunhos de anotação e overlays.
- `scripts/arm_metrics.py`: métricas de qualidade e custo.
- `scripts/viewer_check.mjs`: inspeção do Viewer.
- `logs/`: JSON e stderr do CLI e `*.cost.json` (tempo de parede e pico de GPU por processo).
- `results/`: `metrics.json`, `summary-table.md` e `viewer/`.
