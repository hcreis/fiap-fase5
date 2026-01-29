"""
Cloud Architecture Detector - Inference
Script para usar o modelo treinado em novas imagens
"""

from pathlib import Path
from typing import List, Dict
import json
import argparse
from ultralytics import YOLO
import torch

class CloudArchitectureInference:
    """Realiza inferência com o modelo treinado"""
    
    def __init__(self, model_path: Path):
        """
        Inicializa inferência
        
        Args:
            model_path: Caminho para o diretório do modelo salvo
        """
        # Detectar device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Carregar modelo
        self.load_model(Path(model_path))
        
        # Carregar mapeamento de classes
        self.class_map = {}
        
        # Tentar diferentes locais para classes.txt
        possible_paths = [
            Path(model_path)/'classes.txt',
            Path('./dataset_prepared/classes.txt'),
            Path('./classes.txt'),
        ]
        
        classes_file = None
        for path in possible_paths:
            if path.exists():
                classes_file = path
                break
        
        if classes_file:
            print(f"📚 Carregando classes de: {classes_file}")
            with open(classes_file) as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        idx, name = parts
                        self.class_map[int(idx)] = name
            print(f"✅ {len(self.class_map)} classes carregadas")
        else:
            print("⚠️  Arquivo classes.txt não encontrado")
    
    def load_model(self, model_path: Path):
        """Carrega modelo salvo"""
        detection_weights = model_path / "detection_model.pt"

        if not detection_weights.exists():
            raise FileNotFoundError(f"Pesos não encontrados: {detection_weights}")

        self.detection_model = YOLO(str(detection_weights))
        self.detection_model.to(self.device)

        # Carregar metadata
        metadata_file = model_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                self.provider = metadata.get("provider")
                self.model_size = metadata.get("model_size", "m")
        else:
            self.provider = None
            self.model_size = "m"

        print(f"✅ Modelo carregado de: {model_path}")
    
    def detect(self, image_path: Path, apply_nms: bool = True, confidence: float = 0.5) -> Dict:
        """
        Detecta objetos
        
        Args:
            image_path: Caminho da imagem
            apply_nms: Aplicar NMS para remover detecções sobrepostas
            confidence: Confiança mínima
            
        Returns:
            Dicionário com resultados
        """
        # Realizar predição com YOLO
        results = self.detection_model.predict(
            source=str(image_path), 
            conf=confidence, 
            device=self.device, 
            verbose=False
        )
        
        # Processar resultados
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes

            for i in range(len(boxes)):
                detection = {
                    "class_id": int(boxes.cls[i].item()),
                    "confidence": float(boxes.conf[i].item()),
                    "bbox": boxes.xyxy[i].tolist(),
                }
                detections.append(detection)
        
        result = {"image_path": str(image_path), "detections": detections}
        
        # Mapear class_id para nomes
        for detection in result['detections']:
            detection['class_name'] = self.class_map.get(
                detection['class_id'],
                f"Unknown_{detection['class_id']}"
            )
        
        # Aplicar NMS (Non-Maximum Suppression) para remover detecções sobrepostas
        if apply_nms and len(result['detections']) > 1:
            result['detections'] = self._apply_nms(result['detections'], iou_threshold=0.3)
        
        return result
    
    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """
        Aplica NMS para remover detecções sobrepostas
        Mantém a detecção com maior confiança quando há sobreposição
        
        Args:
            detections: Lista de detecções
            iou_threshold: Threshold IoU para considerar sobreposição
            
        Returns:
            Lista de detecções após NMS
        """
        if not detections:
            return detections
        
        # Ordenar por confiança decrescente
        sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        for i, det in enumerate(sorted_dets):
            should_keep = True
            for kept_det in keep:
                iou = self._calculate_iou(det['bbox'], kept_det['bbox'])
                if iou > iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(det)
        
        return keep
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calcula Intersection over Union entre duas bounding boxes
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area

def filter_by_provider(detections: List[Dict], provider: str) -> List[Dict]:
    """
    Filtra detecções por provedor
    
    Args:
        detections: Lista de detecções
        provider: 'aws', 'azure', 'gcp' ou 'all'
        
    Returns:
        Lista filtrada
    """
    if provider == 'all':
        return detections
    
    filtered = []
    prefix = f"{provider}_"
    for det in detections:
        class_name = det.get('class_name', '').lower()
        if class_name.startswith(prefix):
            filtered.append(det)
    
    return filtered

def main():
    parser = argparse.ArgumentParser(
        description='Cloud Architecture Detector - Detecção com filtro por provedor'
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='all',
        choices=['aws', 'azure', 'gcp', 'all'],
        help='Filtrar detecções por provedor (padrão: all)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.10,
        help='Confiança mínima para aceitar detecção (padrão: 0.50)'
    )
    parser.add_argument(
        '--image',
        type=str,
        default='imagem-modelo2.png',
        help='Caminho da imagem para testar'
    )
    
    args = parser.parse_args()
    
    # Usar cloud_detector_model (modelo completo treinado)
    model_dir = Path('./cloud_detector_model')
    
    if not model_dir.exists():
        print(f"❌ Modelo não encontrado em: {model_dir}")
        print("Execute training_pipeline.py primeiro")
        return
    
    # Inicializar inferência
    inference = CloudArchitectureInference(model_dir)
    
    # Testar em uma imagem
    test_image = Path(args.image)
    
    if not test_image.exists():
        print(f"❌ Imagem não encontrada: {test_image}")
        return
    
    print(f"\n🔍 Testando em: {test_image.name}")
    print(f"   Provedor: {args.provider.upper()}")
    print(f"   Confiança mínima: {args.confidence:.0%}\n")
    
    result = inference.detect(test_image, confidence=args.confidence)
    
    # Aplicar filtro de confiança e provedor
    filtered = [d for d in result['detections'] if d['confidence'] >= args.confidence]
    
    print(f"📊 Resultado:")
    print(f"  Total de objetos detectados (bruto): {len(result['detections'])}")
    print(f"  Após filtro (>= {args.confidence:.0%}): {len(filtered)} objetos")
    
    # Agrupar por provedor
    providers_count = {'aws': 0, 'azure': 0, 'gcp': 0}
    for det in filtered:
        class_name = det.get('class_name', '').lower()
        for prov in providers_count:
            if class_name.startswith(f"{prov}_"):
                providers_count[prov] += 1
                break
    
    print(f"\n📊 Distribuição por Provedor:")
    for prov, count in providers_count.items():
        print(f"  {prov.upper()}: {count}")
    
    # Filtrar por provedor selecionado
    if args.provider != 'all':
        filtered = filter_by_provider(filtered, args.provider.lower())
        provider_name = args.provider.upper()
    else:
        provider_name = "Todos"
    
    print(f"\n📋 Detecções {provider_name} (Confiança >= {args.confidence:.0%}):")
    if filtered:
        for i, detection in enumerate(sorted(filtered, key=lambda x: x['confidence'], reverse=True), 1):
            print(f"    {i}. {detection['class_name']}: {detection['confidence']:.2%}")
    else:
        print(f"    Nenhuma detecção {provider_name} encontrada")
    
    # Mostrar todas as detecções se filtro de provedor estiver ativo e houver outras
    if args.provider != 'all':
        others = [d for d in result['detections'] if d['confidence'] >= args.confidence and d not in filtered]
        if others:
            print(f"\n📋 Outras detecções (confiança >= {args.confidence:.0%}):")
            # Agrupar por provedor
            by_provider = {'aws': [], 'azure': [], 'gcp': []}
            for det in others:
                class_name = det.get('class_name', '').lower()
                for prov in by_provider:
                    if class_name.startswith(f"{prov}_"):
                        by_provider[prov].append(det)
                        break
            
            for provider, dets in by_provider.items():
                if dets:
                    print(f"\n  {provider.upper()}:")
                    for detection in sorted(dets, key=lambda x: x['confidence'], reverse=True):
                        print(f"    - [{provider}] {detection['class_name']}: {detection['confidence']:.2%}")

if __name__ == '__main__':
    main()
