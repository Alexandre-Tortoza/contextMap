# Correção de movimento (deskew) em Geometric Mapping

Este documento descreve `src/contextmap/geometric_mapping/motion_correction.py` (estado, evidência e política) e `src/contextmap/geometric_mapping/deskew.py` (o corretor `pose-interpolation-deskew-v1`).

Um scan de LiDAR é adquirido ao longo de um intervalo de tempo não nulo enquanto a plataforma pode estar em movimento. Tratar todos os pontos como capturados em um único instante pode deformar o mapa. Por outro lado, nenhum consumidor pode **assumir** que um scan foi corrigido só porque um certo estimador foi usado.

## Os três estados

| Estado | Significado |
| --- | --- |
| `RAW` | declarado como **não** corrigido para o movimento da plataforma |
| `CORRECTED` | corrigido, com evidência de quem corrigiu, com qual trajetória, configuração e payload |
| `UNKNOWN` | nada foi declarado; é o **padrão** e nunca é inferido |

`UNKNOWN` é honesto: não é `RAW` nem `CORRECTED`. Nenhum campo de `SourceObservation`, tipo de fonte, sensor ou estimador muda o estado. Em particular, **não existe** a suposição `FAST-LIO ⇒ deskewed=true`: um scan cuja proveniência diz `source_type="fast_lio"` continua `UNKNOWN` até que uma declaração explícita o mude. `unknown_motion_correction(observation)` e `declared_raw_motion_correction(observation)` criam os registros; só o segundo afirma algo, e só para fontes que o operador sabe entregar scans crus.

## `MotionCorrectionRecord`

Registro por scan, com identidade preservada (`observation_id` é a observação física de origem; a observação **nunca** é reescrita).

| Campo | Significado |
| --- | --- |
| `observation_id` | observação de origem |
| `state` | `RAW`, `CORRECTED` ou `UNKNOWN` |
| `acquisition_start` / `acquisition_end` | intervalo de aquisição, quando conhecido (ambos ou nenhum, mesmo clock, fim não anterior ao início) |
| `per_point_timing_available` | se o payload carrega tempo por ponto |
| `evidence` | suporte de uma afirmação `CORRECTED`; ausente nos demais estados |

Regras validadas na construção:

- um scan `CORRECTED` **exige** evidência e exige declarar o tempo que usou (intervalo de aquisição ou tempo por ponto). O tempo **nunca é fabricado**: sem ele, a afirmação é recusada;
- evidência em um scan `RAW` ou `UNKNOWN` é recusada: não se pode carregar um argumento de correção em um scan que não a tem;
- o intervalo de aquisição deve ser completo, ordenado e em um único clock.

`verify_motion_correction(record, observation)` confere o registro contra o scan e devolve uma lista de problemas legíveis: observação diferente, clock do scan diferente do clock do intervalo, timestamp do scan fora do intervalo declarado e, para um registro `CORRECTED`, `payload_hash` da evidência diferente do SHA-256 do payload realmente entregue pelo scan (#595; a mensagem cita os dois hashes). Lista vazia significa consistente.

## `MotionCorrectionEvidence`

O que sustenta uma afirmação de correção, para que ela seja rastreável até a origem exata:

- `producer`: implementação/backend que corrigiu (não vazio);
- `trajectory_id`: trajetória (fonte de estado) usada;
- `payload_hash`: `sha256:<64 hex>` do payload corrigido;
- `configuration_fingerprint`: hash da configuração de correção, quando existe.

A evidência não impõe um estimador: o produtor pode ser um backend de State Estimation ou uma implementação dedicada de preparação de geometria. O contrato só define o estado canônico e as regras de integração.

## Política de scans não corrigidos

`MotionCorrectionPolicy(raw=..., unknown=...)` decide, por estado, o que uma execução faz com o scan:

| `ScanDisposition` | Efeito |
| --- | --- |
| `ACCEPT` | usa o scan |
| `WARN` | usa o scan e registra um aviso |
| `REJECT` | deixa o scan de fora e registra o motivo |

Um scan `CORRECTED` é sempre aceito. `raw` e `unknown` **não têm valor padrão**: uma execução não deixa entrar um scan cru ou desconhecido por omissão. `apply_motion_correction_policy(record, policy)` devolve um `MotionCorrectionVerdict` com a disposição e, quando não é `ACCEPT`, uma mensagem que nomeia o scan e o estado.

A disposição "corrigir quando suportado" citada na issue **não existe**. O corretor deste repositório (seção seguinte) é selecionado para a execução inteira, na montagem dos inputs, e não é uma disposição desta política: com ele, todo scan aceito sai `CORRECTED` e um scan que não pode ser corrigido é recusado, nunca mapeado cru no lugar.

## Deskew por interpolação de pose (`pose-interpolation-deskew-v1`)

Um LiDAR rotativo mede cada ponto no seu próprio instante `t_ref + τ_i`, enquanto a plataforma se move. O mapeamento sem deskew coloca o scan inteiro com a pose do timestamp do scan (`t_ref`) e entorta as superfícies pelo movimento durante a aquisição. `deskew.py` remove esse movimento com a própria trajetória: cada ponto é reexpresso no frame do sensor **no instante `t_ref`**:

```text
p' = T_body_source⁻¹ · T_map_body(t_ref)⁻¹ · T_map_body(t_ref + τ_i) · T_body_source · p
```

A cadeia única do mapeamento, `P_map = T_map_body(t_ref) · T_body_source · p'`, passa então a colocar cada ponto com a pose do seu instante (`T_map_body(t_ref + τ_i) · T_body_source · p`). O `TransformTrace` do scan continua válido, e `transformation.py` e `accumulation.py` não mudam. A álgebra é a mesma do `place(..., deskew=True)` de `experiments/semantic-geometry-investigation-20260930/scripts/p3_time_offset_deskew.py`, que coloca o ponto direto no mapa com `T_map_epson(t_header + τ_i) · T_epson_sensor`.

### Política

`PoseInterpolationDeskewPolicy`, sem nenhum valor padrão (nome e unidade do campo de tempo pertencem ao driver do sensor e nunca são herdados de um dataset):

| Campo | Significado |
| --- | --- |
| `time_field` | nome do campo de tempo por ponto no registro do payload (float32 ou float64, `count = 1`) |
| `time_unit` | unidade desse campo (`PointTimeUnit.SECONDS`); o `PointFieldDescriptor` não declara unidade |
| `reference` | de onde os tempos são offsets e em que instante os pontos corrigidos ficam expressos (`PointTimeReference.SCAN_TIMESTAMP`) |
| `time_bin_ns` | largura do bin temporal: os pontos são agrupados pelo múltiplo mais próximo do seu instante (empate para o par) e uma consulta de pose serve o bin inteiro, no múltiplo **limitado ao primeiro e ao último instante dos pontos do bin**; o erro de tempo é no máximo meio bin |

`policy_id` é `pose-interpolation-deskew-v1` e é também o `producer` da evidência. `encode_deskew_policy()` dá a forma JSON da política.

### Regras

- **Mesmas poses, mesmas regras.** Cada bin é resolvido por `TrajectoryLookup.pose_at` com a **mesma** `LookupPolicy` que resolveu a pose do scan (interpolação linear + slerp, mesma tolerância de gap). O deskew exige uma política `INTERPOLATED`: com `EXACT` quase todo instante seria recusado e com `NEAREST` os pontos andariam em degraus. Outra política é erro de configuração (`GeometryInputError` na montagem).
- **Nunca consultar onde nenhum ponto foi medido.** Sem o limite acima, o último bin do scan (τ ≈ 100,84 ms no corridor-02) arredondava para 101 ms, depois da amostra de pose seguinte, e caía num gap onde nenhum ponto existe. Isso recusava dois scans a mais na janela 628–637 s (medido; teste de regressão `test_a_time_bin_is_never_looked_up_after_the_last_instant_it_serves`).
- **Instante sem pose aceitável ⇒ scan recusado.** Se algum bin cai fora da trajetória ou num intervalo entre poses maior que a tolerância, o scan inteiro é recusado com `InputRejectionReason.DESKEW_REJECTED`, e o detalhe nomeia o offset e o motivo do lookup. O scan nunca segue sem correção em silêncio.
- **Tempo por ponto obrigatório.** Um campo de tempo ausente ou que não é um único float, e um ponto com coordenadas finitas e tempo não finito, recusam o scan pelo mesmo motivo. Pontos com coordenadas não finitas ficam intocados; a transformação os descarta e conta, como antes.
- **Identidade exata.** Um ponto cujo bin é consultado em `t_ref`, ou cujo bin tem pose idêntica à de referência, guarda os seus bytes. Com `τ = 0` em todos os pontos, o payload corrigido é idêntico ao bruto.
- **Nunca corrigir duas vezes.** Com deskew selecionado, um scan já declarado `CORRECTED` é recusado (`DESKEW_REJECTED`).
- **Política de scans não corrigidos.** Com deskew, `MotionCorrectionPolicy` não se aplica ao estado de entrada (`RAW`/`UNKNOWN`), porque a execução corrige o scan. O registro corrigido passa pela política e é sempre `ACCEPT`.

### Saída e evidência

`deskew_scan()` devolve um `DeskewedScan`: a **mesma** `LidarObservation` (identidade, frame, timestamp, layout e todos os outros campos, inclusive o tempo por ponto) com só `x, y, z` reescritos no tipo escalar do próprio payload, e um `MotionCorrectionRecord(CORRECTED, per_point_timing_available=True)` com:

- `producer = "pose-interpolation-deskew-v1"`;
- `trajectory_id` da trajetória usada;
- `payload_hash` = SHA-256 do payload **corrigido**. Assim, `verify_motion_correction(record, observação corrigida)` fica vazio e contra o payload bruto acusa a divergência;
- `configuration_fingerprint` = `deskew_configuration_fingerprint(policy, pose_lookup)`, o SHA-256 do JSON canônico da política **e** da `LookupPolicy`. Muda quando qualquer parâmetro de qualquer uma das duas muda.

O registro não declara intervalo de aquisição: o tempo usado é o tempo por ponto, e o intervalo não é fabricado.

### Integração: montagem dos inputs

`assemble_geometry_inputs(..., deskew=policy)` e `assemble_geometry_inputs_from_artifacts(..., deskew=policy)` aplicam o deskew depois da consulta da pose do scan. `GeometryInput.observation` passa a ser o scan corrigido e `GeometryInput.payload_hash` o hash dele. `GeometryInputPlan.deskew` registra a política. Com `deskew=None` (padrão), o caminho é o de antes: a mesma observação, o mesmo hash, o estado declarado e o `geometry.bin` byte a byte idêntico (teste de regressão com hash fixo em `tests/geometric_mapping/test_deskew.py`, medido antes da mudança; no corridor-02 o E0 é reproduzido byte a byte, ver a validação abaixo).

### Semântica de `source xyz` (decisão D2)

Para um scan `CORRECTED`, as coordenadas de fonte do `geometry.bin` (e `GeometryPoint.source_coordinates_m`) significam **coordenadas do sensor corrigidas para `t_ref`**, não a medição crua. O formato do `geometry.bin` não muda. A medição crua continua recuperável na `SequenceArtifact`, pelo `observation_id` e pelo índice do ponto. O `ScanRecord.payload_hash` de um scan corrigido nomeia o payload corrigido; o bruto é identificado pela observação da sequência. As coordenadas corrigidas ficam no tipo escalar do payload: em float32 o arredondamento é ≲ 6·10⁻⁸ relativo (≈ 0,6 µm a 10 m).

### Validação medida (corridor-02, janela 628–637 s)

`experiments/correction-benchmark-10_31-20260930/deskew/` aplica este deskew pelo caminho real de montagem aos scans do run `workspace-old/frame-10_31-raw`, com a mesma métrica ponto-plano da investigação. Números e ressalvas estão no `README.md` e em `results/deskew_validation.json` desse diretório.

## Estado por ponto e serialização

`GeometryPointProvenance.motion_correction` carrega o estado do scan de origem em **cada ponto**, com `UNKNOWN` como padrão, então um leitor a jusante distingue geometria crua de geometria corrigida sem consultar outra estrutura. O campo é aditivo: os contratos de #89 continuam válidos.

`serialization.py` codifica e decodifica o registro (`encode_motion_correction_record`/`decode_motion_correction_record`), a política (`encode_motion_correction_policy`/`decode_motion_correction_policy`) e o estado por ponto (`provenance.motion_correction`), com apenas primitivas JSON. A decodificação revalida as regras acima. Registrar a política e um registro por scan mapeado no manifesto de execução é responsabilidade do `GeometricMapArtifact`, que ainda **não existe**: este módulo só fornece as formas serializáveis.

## Limitações

- O único corretor é `pose-interpolation-deskew-v1`. Ele depende da qualidade e da densidade da trajetória: com a GT a ~10 Hz, entre amostras o movimento é suposto de velocidade constante (lerp + slerp). Deskew contínuo com IMU ou scans já corrigidos por outro estimador (declarados `CORRECTED` com evidência) ficam para depois.
- Um scan cuja cauda cai num intervalo de trajetória maior que a tolerância é recusado inteiro, mesmo quando a pose do seu timestamp é aceita. Sem deskew esse mesmo scan entra, com a cauda colocada pela pose do início.
- Fora do corretor, `per_point_timing_available` é só um indicador declarado por quem corrigiu.
- `GeometryInputPlan.deskew` ainda não entra na configuração persistida nem no fingerprint do `GeometricMapArtifact`, e a evidência por scan (produtor, fingerprint) não é gravada no índice de origem; hoje o artefato registra só o estado `corrected` e o hash do payload corrigido por scan.
