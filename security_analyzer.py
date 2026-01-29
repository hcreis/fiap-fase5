import requests
import json
from typing import Dict, Any, List

class SecurityAnalyzer:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.api_generate = f"{ollama_url}/api/generate"
        self.api_tags = f"{ollama_url}/api/tags"

    def check_connection(self) -> bool:
        """Verifica se o Ollama está acessível"""
        try:
            # Tenta listar models apenas para ver se a API responde
            response = requests.get(self.api_tags, timeout=2)
            return response.status_code == 200
        except:
            return False
            
    def get_available_models(self) -> List[str]:
        """Retorna lista de modelos disponíveis no Ollama"""
        try:
            response = requests.get(self.api_tags, timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
            return []
        except:
            return []

    def generate_stride_report(self, detection_data: Dict[str, Any], model: str = "llama3") -> str:
        """
        Gera um relatório de vulnerabilidades baseado em STRIDE usando Ollama.
        """
        
        # Filtra apenas informações relevantes para não estourar contexto e remove duplicatas de classes para análise geral
        detections = detection_data.get("detections", [])
        
        # Contagem de componentes
        component_counts = {}
        for d in detections:
            name = d["class_name"]
            component_counts[name] = component_counts.get(name, 0) + 1
            
        components_summary = [
            {"componente": name, "quantidade": count, "provedor": name.split('_')[0] if '_' in name else "genérico"}
            for name, count in component_counts.items()
        ]

        if not components_summary:
            return "⚠️ Nenhum componente detectado para análise de segurança."

        prompt = f"""
Atue como um Engenheiro de Segurança de Nuvem Sênior (Cloud Security Architect).
Analise a seguinte arquitetura baseada nos componentes detectados em um diagrama e gere um relatório de ameaças usando a metodologia STRIDE.

**CONTEXTO DA ARQUITETURA:**
Componentes detectados:
{json.dumps(components_summary, indent=2)}

**SUA TAREFA:**
Gerar um relatório técnico de modelagem de ameaças (Threat Modeling Report) contendo:

1. **Análise por Componente (STRIDE):**
   Para CADA UM dos componentes únicos listados acima (analise TODOS, um por um), aplique a metodologia STRIDE:
   - **Componente:** [Nome]
   - **Vulnerabilidade Potencial (STRIDE category):** Descreva o risco principal.
   - **Mitigação Recomendada:** Solução técnica específica (ex: "Habilitar TLS 1.3", "Configurar WAF", "Rotação de chaves KMS").

2. **Análise de Superfície de Ataque:**
   Com base na combinação dos componentes, identifique pontos fracos na integração (ex: Banco de dados exposto publicamente, falta de Load Balancer).

3. **Checklist de Conformidade (Resumo):**
   3 itens essenciais de segurança para esta arquitetura específica.

**DIRETRIZES:**
- Responda em Português do Brasil.
- Use formatação Markdown (negrito, listas, títulos).
- Seja prático e direto. Não explique o que é STRIDE, apenas aplique-o.
- Se houver componentes de provedores mistos (AWS/Azure), alerte sobre complexidade multicloud.

Gere o relatório agora.
"""

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3, # Mais determinístico e técnico
                "num_ctx": 4096
            }
        }

        try:
            response = requests.post(self.api_generate, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get("response", "❌ Erro: Resposta vazia do modelo.")
        except requests.exceptions.ConnectionError:
            return f"❌ Erro de Conexão: Não foi possível conectar ao Ollama em {self.ollama_url}. Verifique se ele está rodando (`ollama serve`)."
        except requests.exceptions.Timeout:
            return "❌ Erro: O modelo demorou muito para responder (Timeout)."
        except Exception as e:
            return f"❌ Erro ao gerar relatório: {str(e)}"
