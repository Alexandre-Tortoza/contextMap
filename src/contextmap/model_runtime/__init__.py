"""Infraestrutura de execução de modelos compartilhada pelos adapters.

Aqui mora o que é mecânica de rodar um modelo já carregado — fixar qual versão dele
responde, preparar os tensores de entrada, conferir onde seus pesos estão e devolver a
memória quando o estágio termina. Nada disso é domínio: não há região, feature, claim
nem score neste pacote, e ele não importa capability alguma.

Existe porque adapters de ``region_perception``, ``visual_features`` e
``semantic_interpretation`` precisam exatamente da mesma mecânica, e ela estava escondida
sob caminhos privados de ``visual_perception/adapters``. Um pacote declarado torna a
dependência explícita e permitida por declaração, não por omissão da regra de boundary.

Não confundir com :mod:`contextmap.runtime`, o composition root, que resolve
configuração, escolhe backends e executa o grafo de estágios. Este pacote não escolhe
nada: ele é chamado pelo adapter que já foi escolhido.

Os módulos se separam pela dependência externa que cada um isola, porque é ela que lhes
dá motivos independentes de mudança:

```text
huggingface      -> identidade de revisão e pré-processamento de pixels do Hugging Face
model_placement  -> inspeção de device e dtype dos tensores de um modelo torch
model_release    -> devolução do modelo e da memória cacheada entre estágios
```
"""

from contextmap.model_runtime.checkpoints import (
    combined_checkpoint_fingerprint,
    resolve_checkpoint_file,
)
from contextmap.model_runtime.embedding_values import validate_and_normalize_feature_values
from contextmap.model_runtime.huggingface import (
    preprocess_pixel_values,
    validate_huggingface_commit_revision,
)
from contextmap.model_runtime.model_placement import names_device, verify_model_placement
from contextmap.model_runtime.model_release import free_cuda_memory, release_runtime

__all__ = [
    "combined_checkpoint_fingerprint",
    "free_cuda_memory",
    "names_device",
    "preprocess_pixel_values",
    "release_runtime",
    "resolve_checkpoint_file",
    "validate_and_normalize_feature_values",
    "validate_huggingface_commit_revision",
    "verify_model_placement",
]
