# Seleção de centros: `observation-support-anchor-v1`

Este documento descreve `src/contextmap/point_representation/center_selection.py`.

O `RepresentationService` representa exatamente os centros que recebe. **Quais** centros são decide o que o run descreve e quanto ele custa, então a escolha é uma política versionada desta capability, gravada no manifest do run (`center_selection_policy_id`). Não é uma regra de quem executa. O runtime só seleciona a política pelo componente `point_representation.center_selection` e a aplica (ver [`runtime/docs/composition.md`](../../runtime/docs/composition.md)).

Hoje existe uma política: `observation-support-anchor-v1` (`ObservationSupportAnchorPolicy`, sem parâmetros).

## A regra

Para cada `SpatialObservation` de um run de associação:

1. o **centróide** é a média das coordenadas autoritativas (`coordinates_m`, em metros, no frame do mapa) dos membros de `geometry_support`, somadas com `math.fsum`;
2. a **âncora** é o membro de `geometry_support` com a menor distância euclidiana até esse centróide; um empate é decidido pelo menor `geometry_id`;
3. uma observação sem `geometry_support` não ancora nada;
4. a mesma âncora, alcançada por várias observações, é **um** centro.

Os centros saem ordenados por `geometry_id`, então o pedido ao serviço, e com ele a identidade de cada representação (`representation_id_for` usa a posição do centro no pedido), não depende da ordem em que as observações são lidas. Uma observação de outro mapa é recusada com `ValueError`, antes de qualquer suporte; um membro ausente do mapa é o `KeyError` da porta `GeometrySource`.

A política lê só a geometria que a associação já declarou visível dentro da região. Ela não lê máscara, claim, feature visual nem label: a associação é contexto de **seleção**, gravado como `association_context_id`, nunca uma fonte de features.

## É uma heurística

A âncora **não** é o centro do objeto, um ponto saliente nem uma amostra representativa do suporte. É um ponto determinístico, real (um elemento do mapa, nunca uma coordenada sintética) e dentro do que a câmera viu. A partir dele o `SupportExtractor` monta a vizinhança sob a política de suporte do encoder.

**Por que ela.** Ela foi escolhida pelo custo e pela correspondência com o que foi observado, não por qualidade medida:

- o custo é limitado pelo **número de observações**, não pelo tamanho do mapa. No corridor-02, 10 frames deram 307 observações espaciais, ou cerca de 300 centros: uns 10 s de PTv3 a ~30 ms por suporte. Representar o mapa inteiro (26,63 M pontos) levaria dias no PTv3, e representar toda a geometria associada (de 3,8 mil a 36 mil pontos por frame) levaria da ordem de uma hora e ~1 GB de artifact (ver [`artifact.md`](artifact.md#decisões-e-limites));
- cada `ContextRun` representa a geometria que as **suas** observações tocaram, então o contexto incremental cresce com a evidência, sem refazer o mapa inteiro a cada seleção;
- não há parâmetro (raio de voxel, fração de amostragem) cujo valor exigiria uma avaliação própria antes de existir.

**Limites.**

- **Um centro por observação.** Uma região grande, ou que junta superfícies diferentes, tem uma única âncora, e a estrutura longe dela só entra se couber no suporte do encoder. Nada garante que a vizinhança da âncora represente a região inteira.
- **O centróide pode cair fora da superfície.** Numa região côncava ou dividida (duas pernas de uma cadeira), o centróide fica no vazio e a âncora é só o membro mais próximo dele, que pode estar numa borda.
- **Herda os erros da associação.** Um ponto de fundo ou de um oclusor que a associação aceitou dentro da máscara entra no centróide e pode virar a âncora.
- **Âncoras próximas não são deduplicadas.** Observações do mesmo objeto em frames diferentes dão âncoras vizinhas, mas distintas, e os suportes delas se sobrepõem. Só a mesma âncora exata vira um centro. Representações repetidas da mesma estrutura **não** são observações físicas independentes.
- **Custo por membro.** A política resolve cada membro de cada observação por `GeometrySource.get` para achar o centróide, então o custo dela cresce com a geometria associada, ainda que o do encoder cresça só com o número de centros.

Uma regra diferente (mais de uma âncora por observação, uma grade sobre a geometria associada, uma amostragem do mapa) é outra política, com outro `policy_id`. A versão de hoje não muda em silêncio.

## Onde isso aparece

- `PointRepresentationRunManifest.center_selection_policy_id` grava a política. É `None` quando o próprio chamador escolheu os centros, como no harness de avaliação, que fixa os mesmos centros em todos os braços.
- `center_selection_id` continua sendo o hash do **conjunto** de centros pedido, independente da ordem: a política diz como os centros foram escolhidos, e o hash diz quais foram.
