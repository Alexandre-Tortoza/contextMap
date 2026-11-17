"""Encoders concretos, mantidos fora da API pública da capability.

Este pacote **não exporta nada**, e não deve passar a exportar. Um backend é alcançado pelo
caminho do próprio módulo (``backends.geometric_descriptor``, ``backends.ptv3``), e só a partir
do composition root em ``contextmap.runtime``, que é quem escolhe qual encoder roda. Uma fachada
aqui criaria uma segunda superfície pública ao lado de ``contextmap.point_representation`` e
importaria eagerly todo backend futuro, destruindo a garantia de que importar a capability não
carrega biblioteca de modelo.
"""
