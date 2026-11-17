"""Adapters concretos de extração de features visuais.

Cada módulo aqui liga um modelo específico (DINOv2, DINOv3, CLIP, AlphaCLIP, SigLIP2) aos
ports de :mod:`contextmap.visual_perception.visual_features.ports`, e é o único lugar da
capability onde um SDK de modelo pode ser importado.

As torres de texto dos mesmos modelos **não** vivem aqui: elas codificam a hipótese de um
claim para a comparação, o que é scoring, e estão em
:mod:`contextmap.semantic_scoring.adapters`. Cada uma declara o embedding space da torre
visual correspondente, que continua sendo definido neste pacote.

Os adapters ficam fora da API pública: quem escolhe entre eles é o composition root, e
consumidores cross-capability dependem dos ports, não destes módulos.
"""
