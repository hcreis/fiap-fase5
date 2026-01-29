#!/usr/bin/env python3
"""
Cloud Architecture Detector - Streamlit Interface
Interface web tipo chatbot para detecção de componentes cloud
"""

import streamlit as st
from pathlib import Path
import json
import tempfile
from PIL import Image
import sys

# Importar inference e security analyzer
from inference import CloudArchitectureInference
from security_analyzer import SecurityAnalyzer

# Configuração da página
st.set_page_config(
    page_title="Cloud Architecture Detector",
    page_icon="☁️",
    layout="wide"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: #f0f2f6;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1f77b4;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .json-output {
        background-color: #263238;
        color: #aed581;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Inicializar Security Analyzer
if 'security_analyzer' not in st.session_state:
    st.session_state.security_analyzer = SecurityAnalyzer()
    st.session_state.ollama_connected = st.session_state.security_analyzer.check_connection()

if 'model' not in st.session_state:
    with st.spinner('🔄 Carregando modelo...'):
        try:
            model_dir = Path('cloud_detector_model')
            if not model_dir.exists():
                st.error("❌ Modelo não encontrado! Execute start_training.py primeiro.")
                st.stop()
            st.session_state.model = CloudArchitectureInference(model_dir)
            st.success("✅ Modelo carregado com sucesso!")
        except Exception as e:
            st.error(f"❌ Erro ao carregar modelo: {e}")
            st.stop()

# Header
st.markdown('<div class="main-header">☁️ Cloud Architecture Detector</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    
    confidence_threshold = 0.20
    
    provider_filter = st.selectbox(
        "Filtrar por provedor",
        options=["all", "aws", "azure", "gcp"],
        format_func=lambda x: x.upper() if x != "all" else "Todos"
    )
    
    apply_nms = st.checkbox(
        "Aplicar NMS",
        value=True,
        help="Non-Maximum Suppression para remover detecções sobrepostas"
    )
    
    st.markdown("---")
    st.markdown("### 🛡️ Segurança (LLM)")
    
    ollama_status = "✅ Online" if st.session_state.get('ollama_connected') else "❌ Offline"
    st.caption(f"Status do Ollama: {ollama_status}")
    
    selected_model = "llama3"
    if st.session_state.get('ollama_connected'):
        available_models = st.session_state.security_analyzer.get_available_models()
        if available_models:
            selected_model = st.selectbox("Modelo LLM", options=available_models, index=0)
    
    auto_security_scan = st.checkbox("Gerar relatório STRIDE autom.", value=False, disabled=not st.session_state.get('ollama_connected'))

    st.markdown("---")
    
    if st.button("🗑️ Limpar histórico"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Informações")
    st.info(f"""
    **Modelo:** YOLOv8m  
    **Classes:** 104  
    **Device:** CUDA  
    """)

# Área de chat
chat_container = st.container()

with chat_container:
    # Mostrar histórico de mensagens
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f'<div class="chat-message user-message">👤 <b>Você:</b> {message["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            # Exibe o conteúdo da mensagem se houver
            content = message.get("content", "")
            st.markdown(f'<div class="chat-message bot-message">🤖 <b>Detector:</b> {content}</div>', 
                       unsafe_allow_html=True)
            
            # Mostrar imagem se existir
            if 'image' in message:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(message['image'], caption="Imagem analisada", width=400)
                with col2:
                    if message.get('detections'):
                        st.markdown("**📋 Detecções encontradas:**")
                        for i, det in enumerate(message['detections'][:5], 1):
                            st.markdown(f"{i}. **{det['class_name']}** - {det['confidence']:.1%}")
                        if len(message['detections']) > 5:
                            st.markdown(f"*... e mais {len(message['detections']) - 5} detecções*")
                    else:
                        st.warning("Nenhuma detecção encontrada")
            
            # Mostrar JSON
            if 'json_result' in message:
                with st.expander("📄 Ver JSON completo", expanded=False):
                    json_str = json.dumps(message['json_result'], indent=2, ensure_ascii=False)
                    st.code(json_str, language='json')
                
                # Botão para gerar relatório se não for automático
                if st.session_state.get('ollama_connected'):
                    # Identificador único para o botão
                    msg_idx = st.session_state.messages.index(message)
                    if st.button(f"🛡️ Gerar Relatório de Segurança STRIDE", key=f"btn_stride_{msg_idx}"):
                        with st.spinner('🛡️ Gerando análise com Ollama...'):
                             report = st.session_state.security_analyzer.generate_stride_report(
                                 message['json_result'], 
                                 model=selected_model
                             )
                             st.session_state.messages.append({
                                 'role': 'assistant',
                                 'content': report,
                                 'is_report': True
                             })
                             st.rerun()

# Input de imagem
st.markdown("---")
st.markdown("### 📤 Enviar nova imagem")

uploaded_file = st.file_uploader(
    "Escolha uma imagem de arquitetura cloud",
    type=['png', 'jpg', 'jpeg'],
    help="Faça upload de um diagrama de arquitetura cloud"
)

if uploaded_file is not None:
    if st.button("🔍 Analisar imagem", type="primary", use_container_width=True):
        # Adicionar mensagem do usuário
        st.session_state.messages.append({
            'role': 'user',
            'content': f'Enviou: {uploaded_file.name}'
        })
        
        with st.spinner('🔄 Processando imagem...'):
            try:
                # Salvar imagem temporariamente
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp_file.name)
                
                # Realizar detecção
                result = st.session_state.model.detect(
                    tmp_path,
                    apply_nms=apply_nms,
                    confidence=confidence_threshold
                )
                
                # Filtrar por provedor se necessário
                detections = result['detections']
                if provider_filter != 'all':
                    detections = [
                        d for d in detections 
                        if d.get('class_name', '').lower().startswith(f"{provider_filter}_")
                    ]
                
                # Preparar resultado JSON
                json_result = {
                    'success': True,
                    'image': uploaded_file.name,
                    'total_detections': len(detections),
                    'confidence_threshold': confidence_threshold,
                    'provider_filter': provider_filter,
                    'detections': [
                        {
                            'class_name': det['class_name'],
                            'confidence': float(det['confidence']),
                            'bbox': [float(x) for x in det['bbox']],
                            'provider': det['class_name'].split('_')[0] if '_' in det['class_name'] else 'unknown'
                        }
                        for det in detections
                    ]
                }
                
                # Adicionar resposta do bot
                image = Image.open(uploaded_file)
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': f'Análise concluída! Encontradas {len(detections)} detecções.',
                    'image': image,
                    'detections': detections,
                    'json_result': json_result
                })
                
                # Limpar arquivo temporário
                tmp_path.unlink()
                
                # Se processamento automático estiver ativado
                if auto_security_scan and st.session_state.get('ollama_connected'):
                     with st.spinner('🛡️ Gerando relatório de segurança automático...'):
                        report = st.session_state.security_analyzer.generate_stride_report(
                            json_result, 
                            model=selected_model
                        )
                        st.session_state.messages.append({
                            'role': 'assistant',
                            'content': report,
                            'is_report': True
                        })
                
                # Recarregar para mostrar resultados
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Erro ao processar imagem: {e}")
                import traceback
                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>Cloud Architecture Detector v1.0 | YOLOv8m | FIAP Pós-IA 2026</small>
</div>
""", unsafe_allow_html=True)
