# Testes — Region Perception

A suíte focada está em `tests/unit/region_perception/`:

| Arquivo | Garantia |
|---|---|
| `test_region_evidence.py` | Geometria, coordenadas, provenance e invariantes de Region2D |
| `test_artifacts.py` | Fronteira tipada, atomicidade, imutabilidade, inventário e máscaras |
| `test_producer_identity.py` | IDs independentes do run visual, frame correto, hints remapeados e assembly |
| `test_frame_stream.py` | Preparo, constraints sem pixels inline, outcomes e cobertura da seleção |
| `test_regional_execution_streams.py` | Raw response exata, prompts, máscaras refinadas e scores nativos |

As adições determinísticas seguem Red → Green. Identidade, frame streams, execuções regionais,
assembly e cobertura de seleção tiveram testes falhando pela ausência do comportamento antes
da implementação. A normalização científica não mudou: seus 173 goldens devem ser preservados
sem regravação.

As suítes de grounding, refinamento, confidence guard, SAM2, LocateAnything e externalização
de máscaras foram migradas para este owner. Cobrem resposta raw, erros com arquivo/linha,
linhagem de prompts, rejeição de payloads órfãos, liberação de memória e falha de escrita.
Testes de descoberta e normalização preservam os goldens existentes. O teste de arquitetura
verifica imports pela API pública e impede a volta de codecs e identidades regionais a
`visual_perception`.

Use `-p no:cacheprovider --basetemp=<diretório exclusivo>` nas execuções concorrentes.
A validação final do conjunto integrado é `make check`; testes regionais verdes isoladamente
não substituem esse gate.
