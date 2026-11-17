"""Adapters concretos de descoberta, grounding e refinamento de região.

Cada módulo aqui liga um modelo específico (SAM3, Florence-2, LocateAnything) aos ports
de :mod:`contextmap.region_perception.ports`, e é o único lugar da capability onde um SDK
de modelo pode ser importado.

Os adapters ficam fora da API pública: quem escolhe entre eles é o composition root, e
consumidores cross-capability dependem dos ports, não destes módulos.
"""
