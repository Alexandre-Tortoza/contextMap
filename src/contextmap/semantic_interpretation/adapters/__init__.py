"""Adapters concretos de interpretação semântica.

Cada módulo aqui liga um modelo específico (Qwen, Gemini, Eagle-2.5, Florence-2) ao port
``SemanticInterpreter`` de :mod:`contextmap.semantic_interpretation.ports`,
e é o único lugar da capability onde um SDK de modelo ou cliente remoto pode ser importado.

Os adapters ficam fora da API pública: quem escolhe entre eles é o composition root, e
consumidores cross-capability dependem do port, não destes módulos.
"""
