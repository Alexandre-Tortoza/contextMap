"""Borda ROS da ingestion, composta por adapters de bag e decoders especializados.

Os módulos deste pacote isolam rosbags e as estruturas de mensagens ROS, convertendo-as para os
contratos canônicos da capability. Regras de timestamp, sincronização, seleção e persistência não
devem ser incorporadas aos decoders de modalidade.
"""
