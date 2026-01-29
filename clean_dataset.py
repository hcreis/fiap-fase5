#!/usr/bin/env python3
"""
Remove imagens com anotações incompletas (cobertura baixa)
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import shutil
from PIL import Image

def calculate_coverage(png_path, xml_path, threshold=800):
    """
    Estima cobertura de anotações comparando área de objetos vs tamanho da imagem
    
    Heurística: se a soma das áreas anotadas < 20% da imagem, cobertura baixa
    """
    try:
        # Carregar dimensões da imagem
        img = Image.open(png_path)
        total_area = img.width * img.height
        
        # Carregar objetos anotados
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = root.findall('object')
        
        if not objects:
            return 0.0  # Sem anotações = 0%
        
        # Calcular área total das bounding boxes
        annotated_area = 0
        for obj in objects:
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            annotated_area += (xmax - xmin) * (ymax - ymin)
        
        # Cobertura: % da imagem que foi anotada
        coverage = (annotated_area / total_area) * 100
        return coverage
        
    except Exception as e:
        print(f"   ⚠️  Erro ao processar {png_path.name}: {e}")
        return -1

def remove_low_quality_images(min_coverage=15, dry_run=True):
    """
    Remove imagens com baixa cobertura de anotações
    
    min_coverage: porcentagem mínima de cobertura (padrão 15%)
    dry_run: se True, apenas lista; se False, deleta
    """
    
    dataset_dir = Path('./dataset')
    pngs = sorted(dataset_dir.glob('*.png'))
    
    print(f"\n🔍 ANALISANDO DATASET")
    print(f"   Total de imagens: {len(pngs)}")
    print(f"   Threshold de cobertura: {min_coverage}%\n")
    print("-" * 90)
    
    to_remove = []
    
    for i, png in enumerate(pngs, 1):
        xml = png.with_suffix('.xml')
        coverage = calculate_coverage(png, xml)
        
        status = "✅" if coverage >= min_coverage else "❌"
        
        if (i % 50 == 0) or (coverage < min_coverage):
            print(f"{i:4d}. {png.name[:50]:50s} | Cobertura: {coverage:6.2f}% {status}")
        
        if coverage < min_coverage and coverage >= 0:
            to_remove.append((png, xml, coverage))
    
    print("\n" + "=" * 90)
    print(f"📊 RESULTADO:")
    print(f"   Imagens para remover: {len(to_remove)}")
    print(f"   Imagens para manter: {len(pngs) - len(to_remove)}")
    print(f"   Redução: {len(to_remove)/len(pngs)*100:.1f}%")
    
    if to_remove:
        print(f"\n⚠️  IMAGENS A REMOVER:")
        for png, xml, cov in to_remove[:10]:
            print(f"   • {png.name:50s} ({cov:.1f}% cobertura)")
        if len(to_remove) > 10:
            print(f"   ... e {len(to_remove)-10} mais")
    
    # Confirmar remoção
    if dry_run:
        print(f"\n💡 Modo SIMULAÇÃO (dry-run)")
        print(f"   Execute com --execute para realmente remover os arquivos")
        return False
    else:
        print(f"\n🗑️  REMOVENDO ARQUIVOS...")
        removed_count = 0
        for png, xml, _ in to_remove:
            try:
                png.unlink()
                xml.unlink()
                removed_count += 1
            except Exception as e:
                print(f"   ⚠️  Erro ao remover {png.name}: {e}")
        
        print(f"   ✅ {removed_count} pares de arquivo removidos com sucesso!")
        print(f"   Dataset reduzido para {len(pngs) - removed_count} imagens")
        return True

if __name__ == '__main__':
    import sys
    
    # Modo simulação por padrão
    execute = '--execute' in sys.argv
    
    if not execute:
        print("⚠️  MODO SIMULAÇÃO - Nenhum arquivo será removido")
    else:
        print("🚨 MODO EXECUÇÃO - Arquivos SERÃO removidos!")
    
    remove_low_quality_images(min_coverage=15, dry_run=not execute)
    
    if not execute:
        print("\n💾 Para remover de verdade, execute:")
        print("   python3 clean_dataset.py --execute")
