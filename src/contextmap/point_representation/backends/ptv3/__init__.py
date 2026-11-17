"""Backend PTv3 opcional: apenas a superfície do adaptador é reexportada aqui.

Este pacote expõe somente o que :mod:`~contextmap.point_representation.backends.ptv3.adapter`
declara (configuração, resultado, erros, porta de runtime, telemetria e o encoder). Nada além
disso é reexportado: o runtime real (``PointceptPTv3Runtime``), a arquitetura do backbone, a
voxelização e os auxiliares de checkpoint são alcançados pelo caminho do próprio módulo.

Importar este pacote importa apenas ``adapter``, que usa só a biblioteca padrão; por isso o
backend continua opcional e não carrega numpy, torch, spconv, torch_scatter nem Pointcept.
"""

from contextmap.point_representation.backends.ptv3.adapter import (
    PTV3_ADAPTER_VERSION,
    PTv3Config,
    PTv3Inference,
    PTv3OutOfMemoryError,
    PTv3PointEncoder,
    PTv3Runtime,
    PTv3RuntimeUnavailableError,
    PTv3Telemetry,
)

__all__ = [
    "PTV3_ADAPTER_VERSION",
    "PTv3Config",
    "PTv3Inference",
    "PTv3OutOfMemoryError",
    "PTv3PointEncoder",
    "PTv3Runtime",
    "PTv3RuntimeUnavailableError",
    "PTv3Telemetry",
]
