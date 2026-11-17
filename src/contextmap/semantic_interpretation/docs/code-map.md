# Mapa do código

| Responsabilidade | Implementação |
| --- | --- |
| Claims, hipóteses, atributos e proveniência de inferência | `semantic_claims.py` |
| Contexto de cena imutável e validação de ownership | `scene_context.py` |
| Requests, capabilities do interpretador e views referenciadas | `interpretation_request.py` |
| Views de pixels, máscara, crops e contornos | `evidence_views.py` |
| Políticas de seleção de views, prompts, orçamento e condicionamento | `interpretation_policy.py` |
| Instruções versionadas, renderização e parsing estruturado | `prompt_policy.py` |
| Montagem/execução de requests e publicação imediata dos desfechos | `service.py` |
| Execuções, falhas de parse, falhas de chamada e diagnósticos de entrada visual | `interpretation_execution.py` |
| Proveniência e codecs das evidências semânticas | `serialization.py` |
| Observações, manifest, writer e reader de artifacts | `run_artifact.py` |
| Auditoria humana e redação de segredos | `interpretation_audit.py` |
| Port substituível de interpretação | `ports.py` |
| Runtimes/SDKs isolados de Qwen, Gemini, Eagle-2.5 e Florence-2 | `adapters/` |

A raiz pública resolve símbolos sob demanda, com tipos explícitos para o type checker. Ler
contratos ou um artifact não carrega SDKs nem modelos. Consumidores importam pela raiz pública;
backend concreto é escolhido pelo composition root de runtime.
