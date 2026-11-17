# Aplicação da percepção visual

A execução visual é composta pelo runtime através de `VisualPerceptionExecutor`.
O executor recebe factories de extratores densos e regionais, seleção regional explícita,
política de preparação e agendamento. Não há serviço monolítico que produza regiões,
features e claims num único resultado persistido.

O executor regional publica as regiões finais; o executor semântico interpreta essas
regiões numa execução independente. O executor visual publica `VisualEvidenceArtifact`.
Consumidores podem construir `PerceptionResult` como view de uma seleção explícita.

Veja [execução](pipeline.md), [artifact visual](visual-evidence-artifact.md) e
[composição global](../../../../docs/runtime-composition.md).
