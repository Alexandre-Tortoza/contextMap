# Comparação semântica e temporal

Dois canais opcionais do par de entidades, ambos independentes e inspecionáveis pela política de resolução: `compare_semantics` (`SemanticEvidence`) e `compare_temporal` (`TemporalEvidence`). Nenhum decide um merge, e a **falta** de evidência semântica ou temporal nunca é evidência negativa. Como os demais canais, cada um é *medido* ou *indisponível*, com status `supporting`, `conflicting`, `neutral` ou `unavailable`.

## Semântica (`entity-semantic-compatibility-v1`)

Uma entidade guarda toda hipótese, alternativa, conflito, abstenção e atributo. O canal compara **tudo isso**, não um label por entidade, e a linha de base é conservadora e **sem ontologia**.

### Relação entre labels

| Relação | Regra |
| --- | --- |
| `same` | iguais depois da normalização (só caixa, pontuação e espaçamento: `Wooden-Pallet` = `wooden pallet`); nada de stemming, lematização ou tradução |
| `refinement` | um label é o outro mais palavras à esquerda que a **política declara** como modificadores, com o núcleo do nome intacto: `wooden pallet` refina `pallet` se `wooden` foi declarado; `pallet rack` **não** é um `pallet` e `toy pallet` não é refinamento se `toy` não foi declarado |
| `related` | só quando a política declara o par; nada é inferido por senso comum |
| `different` | qualquer outro caso |

`SemanticCompatibilityPolicy(refinement_modifiers, related_label_pairs, identity_attributes)` não tem valores padrão para os modificadores (`related_label_pairs` e `identity_attributes` são vazios por padrão): quais palavras são modificadores e quais labels são relacionados são escolha do perfil, validada (normalizados, ordenados e únicos) e entra no fingerprint. `BASELINE_REFINEMENT_MODIFIERS` é um conjunto inicial de cores, materiais e tamanhos que um perfil pode passar; é convenção, não ontologia. `compare_labels` expõe a relação e a regra versionada que a decidiu.

`SemanticMeasurement` guarda o estado de ambiguidade e o número de hipóteses de cada lado, a relação de **cada par** de labels (primária ou alternativa, com a regra) e as comparações de atributos. Nenhuma alternativa é descartada.

### Regras

| Regra | `supporting` | `conflicting` | `neutral` |
| --- | --- | --- | --- |
| `label-compatibility` | algum par é `same` ou `refinement` (o par primária-primária tem precedência no detalhe) | os dois estados são `unambiguous` e nenhum par é compatível nem relacionado (`pallet` ↔ `person`) | só há pares `related`; ou nenhum par compatível mas uma entidade é ambígua ou conflitante: as alternativas continuam abertas e **não é um falso conflito** |
| `attribute-compatibility` (só se há atributo com o mesmo nome dos dois lados) | nunca | algum valor difere, exceto os valores só **declarados pelos claims** (`claim-attribute-v1`), que só conflitam se o nome estiver em `identity_attributes` e as duas entidades não compartilharem nenhum valor declarado | os valores concordam: concordar num material ou numa cor **não** é evidência de identidade; ou só valores declarados pelos claims divergem, sem declaração de relevância para identidade |

Um conflito semântico não é prova de identidade distinta: a semântica upstream pode estar errada. Uma entidade **sem hipótese suportada** (desconhecida ou abstida) torna o canal `unavailable(missing_evidence)`, e o detalhe nomeia a entidade.

Os **scores das hipóteses nunca são lidos**: são sinais heterogêneos sem semântica comum definida, e uma hipótese só existe porque ao menos uma claim a apoia. O atributo `class` (derivado, que só repete a hipótese primária) e o conhecimento externo (nunca é evidência) ficam de fora. Cada comparação de atributo guarda a origem (`observed` ou `derived`) dos dois valores.

### Atributos declarados pelos claims (decisão D6)

Com a propagação `claim-attribute-v1` (C7a), uma entidade carrega o que o interpretador **disse** sobre cada vista: `category`, `color`, `material`, uma `description` livre. Divergir aí é esperado entre observações do mesmo objeto (azul sob uma luz, cinza sob outra; duas descrições de frases diferentes), então tratar a divergência como conflito puniria justamente as entidades que mais precisam de fusão multi-vista (C8). Medido antes da mudança, num teste: `door` com `color=blue` × `door` com `color=gray` dava `attribute-compatibility = conflicting`.

A regra, dentro de `entity-semantic-compatibility-v1`:

- toda comparação continua **registrada** em `SemanticMeasurement.attribute_comparisons` (nada é descartado);
- uma divergência que repousa num valor só declarado por claims é no máximo `neutral`;
- só um nome declarado em `SemanticCompatibilityPolicy.identity_attributes` (minúsculo, ordenado, único; vazio por padrão) pode conflitar, e só quando os valores declarados das duas entidades para aquele nome **não se intersectam** (normalizados como os labels): duas entidades que compartilham `blue` entre as cores declaradas não conflitam mesmo que uma também tenha `gray`;
- atributos de qualquer outra derivação mantêm a regra de antes (valor diferente conflita).

Sem `identity_attributes` o fingerprint da política é byte a byte o de antes, e nenhum run anterior muda: atributos `claim-attribute-v1` não existiam. Escolher quais atributos são relevantes para identidade é decisão de perfil que depende de avaliação (C8), não deste módulo.

## Temporal (`entity-temporal-compatibility-v1`)

Compara o histórico de observações das duas entidades. Frames físicos e resultados de inferência são contados **à parte**: várias inferências sobre o mesmo frame são correlacionadas, nunca vistas independentes.

`TemporalMeasurement` guarda o domínio de relógio, a sobreposição **ou** a lacuna entre os intervalos (em ns; intervalos que se tocam têm as duas em 0), os frames físicos e os resultados de inferência de cada lado, os frames compartilhados e a união de frames distintos. Repetir a inferência não aumenta a união.

| Regra | `conflicting` | `neutral` |
| --- | --- | --- |
| `static-scene-co-observation` | a política declara `static_scene=True` **e** as entidades compartilham pelo menos um frame físico: numa cena estática, um objeto não aparece duas vezes num frame | senão (inclusive sem cena estática declarada) |

`TemporalCompatibilityPolicy(static_scene)` **não tem padrão**: a suposição de cena estática precisa ser justificada pelo perfil, porque uma segmentação excessiva pode dividir um objeto em duas regiões do mesmo frame. O achado guarda sempre a métrica (`shared_physical_observation_count`) e, com cena estática, o limiar.

O tempo **nunca infere movimento, re-identificação ou desaparecimento**: uma lacuna grande é medida, não julgada, e a premissa é cena estática ou quase estática. Domínios de relógio diferentes tornam o canal `unavailable(incompatible_domain)`, porque os instantes não são comparáveis. Um timestamp ausente não ocorre: uma entidade sem histórico não é uma entidade válida.

## O que estes módulos não fazem

Nenhuma decisão `MATCH`/`DISTINCT`, nenhum LLM ou senso comum, nenhum rastreamento de objetos dinâmicos e nenhuma alteração do estado semântico da entidade: as alternativas e os conflitos permanecem na entidade de origem.
