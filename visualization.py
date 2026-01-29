"""
Cloud Architecture Detector - Visualization & Analysis
Script para visualizar resultados de detecção
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from PIL import Image

class DetectionVisualizer:
    """Classe para visualizar resultados de detecção"""
    
    def __init__(self, class_map: Dict[int, str] = None):
        """
        Inicializa visualizador
        
        Args:
            class_map: Dicionário mapeando class_id para nomes
        """
        self.class_map = class_map or {}
        
        # Cores por classe (BGR para OpenCV)
        self.colors = {
            'aws': (0, 255, 0),      # Verde
            'azure': (255, 0, 0),    # Azul
            'gcp': (0, 165, 255),    # Laranja
            'other': (128, 128, 128) # Cinza
        }
    
    def draw_detections(self, image_path: Path, detections: List[Dict], 
                       output_path: Path = None) -> np.ndarray:
        """
        Desenha detecções na imagem
        
        Args:
            image_path: Caminho da imagem
            detections: Lista de detecções
            output_path: Caminho para salvar (opcional)
            
        Returns:
            Imagem com detecções desenhadas
        """
        image = cv2.imread(str(image_path))
        
        for detection in detections:
            x1, y1, x2, y2 = [int(x) for x in detection['bbox']]
            class_name = detection.get('class_name', 'Unknown')
            confidence = detection.get('confidence', 0)
            
            # Escolher cor baseado na classe
            color = self._get_color(class_name)
            
            # Desenhar bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Desenhar texto
            label = f"{class_name}: {confidence:.2%}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1 - label_size[1] - baseline),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - baseline),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if output_path:
            cv2.imwrite(str(output_path), image)
            print(f"✅ Imagem salva em: {output_path}")
        
        return image
    
    def _get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Define cor baseado no nome da classe"""
        class_name_lower = class_name.lower()
        
        if 'aws' in class_name_lower:
            return self.colors['aws']
        elif 'azure' in class_name_lower:
            return self.colors['azure']
        elif 'gcp' in class_name_lower:
            return self.colors['gcp']
        else:
            return self.colors['other']
    
    def plot_detection_results(self, image_path: Path, detections: List[Dict],
                              similar_images: List[Dict] = None):
        """
        Plota detecções e imagens similares
        
        Args:
            image_path: Caminho da imagem principal
            detections: Detecções encontradas
            similar_images: Imagens similares (RAG)
        """
        # Carregar imagem principal
        main_image = Image.open(image_path)
        
        # Contar número de subplots
        n_cols = 4
        n_similar = len(similar_images) if similar_images else 0
        n_rows = (n_similar + n_cols - 1) // n_cols + 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        
        # Plot imagem principal com detecções
        ax = axes[0]
        image_with_detections = self.draw_detections(image_path, detections)
        image_with_detections_rgb = cv2.cvtColor(image_with_detections, cv2.COLOR_BGR2RGB)
        ax.imshow(image_with_detections_rgb)
        ax.set_title(f"Detecções ({len(detections)})", fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Tabela de detecções
        ax = axes[1]
        ax.axis('off')
        
        if detections:
            table_data = []
            for i, det in enumerate(detections[:5], 1):
                table_data.append([
                    det.get('class_name', 'N/A')[:20],
                    f"{det.get('confidence', 0):.2%}"
                ])
            
            if len(detections) > 5:
                table_data.append([f"... +{len(detections) - 5} mais", ""])
            
            table = ax.table(cellText=table_data, colLabels=['Classe', 'Confiança'],
                           cellLoc='center', loc='center',
                           colWidths=[0.6, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
        else:
            ax.text(0.5, 0.5, 'Nenhuma detecção', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
        
        # Plotar imagens similares
        if similar_images:
            for i, similar in enumerate(similar_images):
                ax = axes[2 + i]
                try:
                    img = Image.open(similar['path'])
                    ax.imshow(img)
                    ax.set_title(f"Sim: {similar.get('similarity', 0):.2%}",
                               fontsize=10)
                except:
                    ax.text(0.5, 0.5, 'Imagem não encontrada',
                           ha='center', va='center',
                           transform=ax.transAxes)
                ax.axis('off')
        
        # Limpar subplots vazios
        for ax in axes[2 + n_similar:]:
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_class_distribution_chart(self, detections_batch: List[Dict]):
        """
        Cria gráfico de distribuição de classes
        
        Args:
            detections_batch: Batch de resultados de detecção
        """
        class_counts = {}
        
        for result in detections_batch:
            for detection in result.get('detections', []):
                class_name = detection.get('class_name', 'Unknown')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Criar gráfico
        fig, ax = plt.subplots(figsize=(12, 6))
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        colors_list = [self._get_color(c) for c in classes]
        colors_list = [(b/255, g/255, r/255) for b, g, r in colors_list]  # BGR to RGB
        
        bars = ax.barh(classes, counts, color=colors_list, edgecolor='black')
        
        # Adicionar valores nas barras
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{int(width)}', ha='left', va='center', fontweight='bold')
        
        ax.set_xlabel('Contagem', fontsize=12, fontweight='bold')
        ax.set_title('Distribuição de Detecções por Classe', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig

class AnalysisReport:
    """Gera relatórios de análise"""
    
    @staticmethod
    def generate_summary(detections_batch: List[Dict]) -> Dict:
        """
        Gera resumo das detecções
        
        Args:
            detections_batch: Batch de resultados
            
        Returns:
            Dicionário com estatísticas
        """
        total_images = len(detections_batch)
        total_detections = sum(len(r.get('detections', [])) for r in detections_batch)
        
        class_stats = {}
        confidence_scores = []
        
        for result in detections_batch:
            for detection in result.get('detections', []):
                class_name = detection.get('class_name', 'Unknown')
                confidence = detection.get('confidence', 0)
                
                if class_name not in class_stats:
                    class_stats[class_name] = {
                        'count': 0,
                        'avg_confidence': [],
                        'min_confidence': 1.0,
                        'max_confidence': 0.0
                    }
                
                class_stats[class_name]['count'] += 1
                class_stats[class_name]['avg_confidence'].append(confidence)
                class_stats[class_name]['min_confidence'] = min(
                    class_stats[class_name]['min_confidence'], confidence
                )
                class_stats[class_name]['max_confidence'] = max(
                    class_stats[class_name]['max_confidence'], confidence
                )
                
                confidence_scores.append(confidence)
        
        # Calcular médias
        for class_name in class_stats:
            scores = class_stats[class_name]['avg_confidence']
            class_stats[class_name]['avg_confidence'] = np.mean(scores) if scores else 0
        
        return {
            'total_images': total_images,
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / total_images if total_images > 0 else 0,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'class_statistics': class_stats
        }
    
    @staticmethod
    def print_report(summary: Dict):
        """Imprime relatório formatado"""
        print("\n" + "="*60)
        print("📊 RELATÓRIO DE ANÁLISE")
        print("="*60)
        
        print(f"\n📈 Estatísticas Gerais:")
        print(f"  Total de imagens: {summary['total_images']}")
        print(f"  Total de detecções: {summary['total_detections']}")
        print(f"  Média por imagem: {summary['avg_detections_per_image']:.2f}")
        print(f"  Confiança média: {summary['avg_confidence']:.2%}")
        
        print(f"\n📋 Por Classe:")
        for class_name, stats in sorted(summary['class_statistics'].items()):
            print(f"\n  {class_name}:")
            print(f"    Detecções: {stats['count']}")
            print(f"    Confiança média: {stats['avg_confidence']:.2%}")
            print(f"    Intervalo: {stats['min_confidence']:.2%} - {stats['max_confidence']:.2%}")

def main():
    """Exemplo de uso"""
    from inference import CloudArchitectureInference
    
    model_dir = Path('cloud_detector_model')
    
    if not model_dir.exists():
        print(f"❌ Modelo não encontrado em: {model_dir}")
        print("Execute start_training.py primeiro")
        return
    
    # Inicializar inferência
    inference = CloudArchitectureInference(model_dir)
    visualizer = DetectionVisualizer()
    
    # Exemplo: Processar uma imagem
    image_path = Path('imagem-modelo.png')
    
    if not image_path.exists():
        print(f"❌ Imagem não encontrada: {image_path}")
        print("Use: python visualization.py")
        return
    
    print("Visualizando resultados de detecção...")
    
    # Realizar detecção (sem RAG)
    result = inference.detect(image_path, confidence=0.5)
    detections = result['detections']
    
    # Visualizar apenas detecções (sem imagens similares)
    output_path = Path('output_detection.png')
    visualizer.draw_detections(image_path, detections, output_path)
    
    print(f"\n📊 Total de detecções: {len(detections)}")
    for i, det in enumerate(detections, 1):
        print(f"  {i}. {det['class_name']}: {det['confidence']:.2%}")

if __name__ == '__main__':
    main()
