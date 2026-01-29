"""
Cloud Architecture Detector - RAG + Object Detection Model
Sistema para detectar e reconhecer ícones de arquitetura cloud em imagens
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class BBoxAnnotation:
    """Classe para representar anotações de bounding box"""
    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    
    def to_yolo_format(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Converte para formato YOLO (center_x, center_y, width, height) normalizado"""
        x_center = ((self.xmin + self.xmax) / 2) / img_width
        y_center = ((self.ymin + self.ymax) / 2) / img_height
        width = (self.xmax - self.xmin) / img_width
        height = (self.ymax - self.ymin) / img_height
        return x_center, y_center, width, height

class CloudArchitectureDataset:
    """Gerencia o dataset de arquitetura cloud"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.images_info = {}
        self.class_names = set()
        self.class_map = {}
        
    def parse_xml_annotation(self, xml_path: Path) -> Tuple[int, int, List[BBoxAnnotation]]:
        """
        Parse arquivo XML de anotação PASCAL VOC
        Retorna: (width, height, lista de anotações)
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        
        annotations = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            
            annotation = BBoxAnnotation(
                name=name,
                xmin=int(bbox.find('xmin').text),
                ymin=int(bbox.find('ymin').text),
                xmax=int(bbox.find('xmax').text),
                ymax=int(bbox.find('ymax').text)
            )
            
            annotations.append(annotation)
            self.class_names.add(name)
        
        return width, height, annotations
    
    def load_dataset(self, image_dir: Path) -> Dict:
        """Carrega todo o dataset de imagens e anotações XML"""
        dataset = {}
        image_paths = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
        
        print(f"Encontradas {len(image_paths)} imagens")
        
        for img_path in image_paths:
            xml_path = img_path.with_suffix('.xml')
            
            if not xml_path.exists():
                print(f"⚠️  Anotação não encontrada para: {img_path.name}")
                continue
            
            try:
                img = Image.open(img_path)
                width, height, annotations = self.parse_xml_annotation(xml_path)
                
                dataset[str(img_path)] = {
                    'image': img_path,
                    'width': width,
                    'height': height,
                    'annotations': annotations,
                    'classes': [a.name for a in annotations]
                }
            except Exception as e:
                print(f"❌ Erro ao processar {img_path.name}: {e}")
        
        # Criar mapa de classes
        self.class_map = {name: idx for idx, name in enumerate(sorted(self.class_names))}
        
        print(f"\n✅ Dataset carregado: {len(dataset)} imagens")
        print(f"Classes encontradas ({len(self.class_map)}): {sorted(self.class_names)}")
        
        return dataset
    
    def create_yolo_format_annotations(self, dataset: Dict, output_dir: Path, images_output_dir: Path = None):
        """Converte anotações para formato YOLO e copia imagens"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if images_output_dir:
            images_output_dir = Path(images_output_dir)
            images_output_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path, info in dataset.items():
            txt_filename = Path(img_path).stem + '.txt'
            txt_path = output_dir / txt_filename
            
            with open(txt_path, 'w') as f:
                for annotation in info['annotations']:
                    class_id = self.class_map[annotation.name]
                    x_center, y_center, width, height = annotation.to_yolo_format(
                        info['width'], info['height']
                    )
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            # Copiar imagem se output_images_dir foi especificado
            if images_output_dir:
                import shutil
                img_filename = Path(img_path).name
                shutil.copy2(img_path, images_output_dir / img_filename)
        
        print(f"✅ Anotações YOLO criadas em: {output_dir}")
        if images_output_dir:
            print(f"✅ Imagens copiadas para: {images_output_dir}")
    
    def create_class_mapping_file(self, output_path: Path):
        """Cria arquivo de mapeamento de classes"""
        with open(output_path, 'w') as f:
            for class_name, class_id in sorted(self.class_map.items(), key=lambda x: x[1]):
                f.write(f"{class_id} {class_name}\n")
        print(f"✅ Mapeamento de classes salvo em: {output_path}")
    
    def get_statistics(self, dataset: Dict) -> Dict:
        """Retorna estatísticas do dataset"""
        class_counts = defaultdict(int)
        
        for info in dataset.values():
            for class_name in info['classes']:
                class_counts[class_name] += 1
        
        return {
            'total_images': len(dataset),
            'total_classes': len(self.class_names),
            'class_distribution': dict(class_counts),
            'total_objects': sum(class_counts.values())
        }

def main():
    """Exemplo de uso"""
    # Dataset dentro da pasta FASE_5
    data_dir = Path('./dataset')
    output_dir = Path('./dataset_prepared')
    
    if not data_dir.exists():
        print(f"❌ Diretório não encontrado: {data_dir}")
        print("Ajuste o caminho em data_dir no script")
        return
    
    # Inicializar dataset
    dataset_manager = CloudArchitectureDataset(data_dir)
    
    # Carregar dataset
    dataset = dataset_manager.load_dataset(data_dir)
    
    if not dataset:
        print("❌ Nenhuma imagem foi carregada")
        return
    
    # Criar anotações em formato YOLO e copiar imagens
    yolo_dir = output_dir / 'labels'
    images_dir = output_dir / 'images'
    dataset_manager.create_yolo_format_annotations(dataset, yolo_dir, images_dir)
    
    # Salvar mapeamento de classes
    dataset_manager.create_class_mapping_file(output_dir / 'classes.txt')
    
    # Estatísticas
    stats = dataset_manager.get_statistics(dataset)
    print("\n📊 Estatísticas do Dataset:")
    print(f"  Total de imagens: {stats['total_images']}")
    print(f"  Total de classes: {stats['total_classes']}")
    print(f"  Total de objetos: {stats['total_objects']}")
    print(f"\n📈 Distribuição por classe:")
    for class_name, count in sorted(stats['class_distribution'].items()):
        print(f"    {class_name}: {count}")

if __name__ == '__main__':
    main()
