#!/usr/bin/env python3
"""
Detecta ícones/objetos não anotados usando detecção automática
"""

from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from scipy import ndimage
import cv2

def find_unannotated_objects(image_path, xml_path, min_size=30):
    """
    Detecta possíveis objetos não anotados na imagem
    """
    # Carregar imagem
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Converter para HSV para melhor detecção de cores
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Detectar cores saturadas (ícones têm cores vibrantes)
    # Threshold para saturation alta
    lower_sat = np.array([0, 100, 100])
    upper_sat = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_sat, upper_sat)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_objects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filtrar muito pequenos
        if w >= min_size and h >= min_size:
            detected_objects.append({
                'bbox': (x, y, x+w, y+h),
                'area': w*h
            })
    
    # Carregar XMLs anotados
    annotated_bboxes = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            annotated_bboxes.append((xmin, ymin, xmax, ymax))
    except:
        return None
    
    # Comparar: encontrar detectados que não sobrepõem anotados
    unannotated = []
    for det in detected_objects:
        dx_min, dy_min, dx_max, dy_max = det['bbox']
        is_covered = False
        
        for ax_min, ay_min, ax_max, ay_max in annotated_bboxes:
            # Verificar sobreposição
            overlap = not (dx_max < ax_min or ax_max < dx_min or 
                          dy_max < ay_min or ay_max < dy_min)
            if overlap:
                is_covered = True
                break
        
        if not is_covered:
            unannotated.append(det)
    
    return {
        'detected': len(detected_objects),
        'annotated': len(annotated_bboxes),
        'unannotated': len(unannotated),
        'coverage': (len(annotated_bboxes) / len(detected_objects) * 100) if detected_objects else 0
    }

# Analisar dataset
dataset_dir = Path('./dataset')
pngs = sorted(dataset_dir.glob('*.png'))[:100]  # Primeiro 100

print("\n📊 ANALISANDO COBERTURA DE ANOTAÇÕES\n")
print("-" * 80)

problems = []
for i, png in enumerate(pngs, 1):
    xml = png.with_suffix('.xml')
    result = find_unannotated_objects(png, xml)
    
    if result:
        coverage = result['coverage']
        status = "✅" if coverage >= 95 else "⚠️ " if coverage >= 80 else "❌"
        
        print(f"{i:3d}. {png.name[:40]:40s} | Detectados: {result['detected']:2d} | Anotados: {result['annotated']:2d} | Cobertura: {coverage:5.1f}% {status}")
        
        if coverage < 100:
            problems.append((png.name, result))

print("\n" + "=" * 80)
print(f"📈 RESUMO:")
print(f"   Imagens analisadas: {len(pngs)}")
print(f"   Imagens com problemas: {len(problems)}")

if problems:
    print(f"\n⚠️  IMAGENS COM OBJETOS NÃO ANOTADOS:")
    for name, res in problems[:10]:
        gap = res['detected'] - res['annotated']
        print(f"   • {name[:40]:40s} | Faltam ~{gap} objetos ({100-res['coverage']:.0f}% não cobertos)")
    if len(problems) > 10:
        print(f"   ... e {len(problems)-10} mais imagens com problemas")

