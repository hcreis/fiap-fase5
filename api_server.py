"""
Cloud Architecture Detector - Flask API
API REST para detecção e reconhecimento de arquitetura cloud
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import json
import logging
from datetime import datetime
from io import BytesIO
from PIL import Image
import numpy as np

# Importar módulos do projeto
from inference import CloudArchitectureInference
from visualization import DetectionVisualizer, AnalysisReport

# Configuração
app = Flask(__name__)
CORS(app)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações
MODEL_DIR = Path('/home/rc/Documents/fiap/Conteudo-PosIA/FASE_5/cloud_detector_model')
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Variáveis globais
inference = None
visualizer = None
request_count = 0

def init_models():
    """Inicializa modelos"""
    global inference, visualizer
    
    logger.info("Inicializando modelos...")
    
    if not MODEL_DIR.exists():
        logger.error(f"Modelo não encontrado em {MODEL_DIR}")
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_DIR}")
    
    try:
        inference = CloudArchitectureInference(MODEL_DIR)
        visualizer = DetectionVisualizer()
        logger.info("✅ Modelos carregados com sucesso")
    except Exception as e:
        logger.error(f"Erro ao carregar modelos: {e}")
        raise

def allowed_file(filename):
    """Verifica extensão de arquivo"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.before_request
def before_request():
    """Middleware antes de cada requisição"""
    global request_count
    request_count += 1
    logger.info(f"Requisição #{request_count}: {request.method} {request.path}")

@app.route('/health', methods=['GET'])
def health():
    """Health check da API"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': inference is not None,
        'requests': request_count
    })

@app.route('/info', methods=['GET'])
def info():
    """Informações sobre o modelo"""
    return jsonify({
        'name': 'Cloud Architecture Detector',
        'version': '1.0',
        'model': 'YOLOv8 + CLIP RAG',
        'endpoints': {
            '/health': 'Health check',
            '/info': 'Informações do modelo',
            '/detect': 'Detectar em imagem única',
            '/batch': 'Processar lote de imagens',
            '/retrieve': 'Recuperar imagens similares',
            '/visualize': 'Gerar visualização'
        },
        'models': {
            'detection': 'YOLOv8 (Object Detection)',
            'retrieval': 'CLIP (Vision-Language Model)',
            'rag': 'Vector Search com Embeddings'
        }
    })

@app.route('/detect', methods=['POST'])
def detect():
    """
    Detecta objetos em uma imagem
    
    Métodos:
    - file: Upload de arquivo
    - url: URL da imagem (não implementado)
    
    Returns:
        JSON com detecções e imagens similares
    """
    try:
        # Validação
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo foi enviado'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Arquivo sem nome'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Tipo de arquivo não permitido. Use: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        if file.content_length > MAX_FILE_SIZE:
            return jsonify({
                'error': f'Arquivo muito grande (máx {MAX_FILE_SIZE / 1024 / 1024}MB)'
            }), 400
        
        # Processar imagem
        logger.info(f"Processando arquivo: {file.filename}")
        
        image_bytes = file.read()
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Salvar temporariamente
        temp_path = Path('/tmp') / f"upload_{datetime.now().timestamp()}.png"
        image.save(temp_path)
        
        # Detector
        result = inference.detect_and_retrieve(temp_path)
        
        # Remover arquivo temporário
        temp_path.unlink()
        
        logger.info(f"Detecções encontradas: {len(result['detections'])}")
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'filename': file.filename,
            'file_size': len(image_bytes),
            'image_size': image.size,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Erro na detecção: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch():
    """
    Processa lote de imagens
    
    Esperado: JSON com lista de caminhos
    {
        "images": ["/path/to/image1.png", "/path/to/image2.png"]
    }
    
    Returns:
        JSON com resultados de todas as imagens
    """
    try:
        data = request.get_json()
        
        if not data or 'images' not in data:
            return jsonify({'error': 'JSON deve conter "images"'}), 400
        
        image_paths = data['images']
        
        if not isinstance(image_paths, list):
            return jsonify({'error': '"images" deve ser uma lista'}), 400
        
        logger.info(f"Processando lote de {len(image_paths)} imagens")
        
        results = []
        errors = []
        
        for i, img_path in enumerate(image_paths, 1):
            try:
                path = Path(img_path)
                
                if not path.exists():
                    errors.append(f"{img_path}: Arquivo não encontrado")
                    continue
                
                logger.info(f"  [{i}/{len(image_paths)}] Processando {img_path}")
                
                result = inference.detect_and_retrieve(path)
                results.append({
                    'image': img_path,
                    'result': result
                })
                
            except Exception as e:
                errors.append(f"{img_path}: {str(e)}")
                logger.error(f"Erro ao processar {img_path}: {e}")
        
        return jsonify({
            'success': len(errors) == 0,
            'timestamp': datetime.now().isoformat(),
            'processed': len(results),
            'total': len(image_paths),
            'errors': len(errors),
            'results': results,
            'error_details': errors if errors else None
        })
        
    except Exception as e:
        logger.error(f"Erro no batch: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrieve', methods=['POST'])
def retrieve():
    """
    Recupera imagens similares (RAG)
    
    Esperado: JSON com imagem
    {
        "file": <arquivo ou base64>,
        "k": 5
    }
    
    Returns:
        Top-K imagens similares
    """
    try:
        k = request.args.get('k', 5, type=int)
        
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo foi enviado'}), 400
        
        file = request.files['file']
        
        # Carregar imagem
        image = Image.open(BytesIO(file.read())).convert('RGB')
        
        # Salvar temporariamente
        temp_path = Path('/tmp') / f"upload_{datetime.now().timestamp()}.png"
        image.save(temp_path)
        
        # Recuperar similares
        similar_images = inference.detector.retrieve_similar_images(temp_path, k=k)
        
        temp_path.unlink()
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'k': k,
            'similar_images': [
                {'path': path, 'similarity': float(sim)}
                for path, sim in similar_images
            ]
        })
        
    except Exception as e:
        logger.error(f"Erro na recuperação: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/visualize', methods=['POST'])
def visualize():
    """
    Gera visualização com detecções
    
    Retorna: Imagem PNG com bounding boxes
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo foi enviado'}), 400
        
        file = request.files['file']
        
        # Carregar imagem
        image = Image.open(BytesIO(file.read())).convert('RGB')
        
        # Salvar temporariamente
        temp_path = Path('/tmp') / f"upload_{datetime.now().timestamp()}.png"
        image.save(temp_path)
        
        # Detecção
        result = inference.detect_and_retrieve(temp_path)
        
        # Visualizar
        img_with_boxes = visualizer.draw_detections(
            temp_path,
            result['detections'],
            output_path=None  # Retornar em memória
        )
        
        temp_path.unlink()
        
        # Converter para PNG
        import cv2
        success, encoded = cv2.imencode('.png', img_with_boxes)
        
        return {
            'data:image/png;base64,' + 
            __import__('base64').b64encode(encoded).decode()
        }
        
    except Exception as e:
        logger.error(f"Erro na visualização: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    """
    Retorna estatísticas do sistema
    """
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'requests_processed': request_count,
        'model_loaded': inference is not None,
        'api_version': '1.0',
        'uptime': 'N/A'  # Implementar com tempo de início
    })

@app.errorhandler(400)
def bad_request(error):
    """Erro 400"""
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400

@app.errorhandler(404)
def not_found(error):
    """Erro 404"""
    return jsonify({'error': 'Endpoint não encontrado', 'message': str(error)}), 404

@app.errorhandler(500)
def internal_error(error):
    """Erro 500"""
    return jsonify({'error': 'Erro interno do servidor', 'message': str(error)}), 500

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Cloud Architecture Detector - Flask API")
    logger.info("=" * 60)
    
    try:
        # Inicializar modelos
        init_models()
        
        # Configurações
        port = 5000
        debug = False
        
        logger.info(f"\n🚀 Iniciando servidor na porta {port}...")
        logger.info(f"📝 Documentação: http://localhost:{port}/info")
        logger.info(f"💓 Health check: http://localhost:{port}/health\n")
        
        # Executar servidor
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug,
            use_reloader=False
        )
        
    except Exception as e:
        logger.error(f"Erro ao iniciar servidor: {e}")
        raise
