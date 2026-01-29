#!/usr/bin/env python3
"""
Analisa problemas nas anotações XML
"""

from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

dataset_dir = Path('./dataset')

# Pegar primeira imagem com api_gateway
api_gateway_imgs = sorted(dataset_dir.glob('*api_gateway*.png'))
test_img = api_gateway_imgs[0] if api_gateway_imgs else sorted(dataset_dir.glob('*.png'))[0]

xml_file = test_img.with_suffix('.xml')

print(f"📊 ANALISANDO: {test_img.name}\n")

# Carregar imagem
img = Image.open(test_img)
print(f"Dimensões da imagem: {img.width}x{img.height}")

# Carregar XML
tree = ET.parse(xml_file)
root = tree.getroot()
objects = root.findall('object')

print(f"Total de objetos no XML: {len(objects)}\n")

# Analisar cada objeto
print("BOUNDING BOXES:")
print("-" * 100)

overlaps = []
for i, obj in enumerate(objects):
    name = obj.find('name').text
    bbox = obj.find('bndbox')
    
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)
    
    width = xmax - xmin
    height = ymax - ymin
    area = width * height
    
    print(f"{i+1:2d}. {name[:35]:35s} | Box: [{xmin:4d}, {ymin:4d}, {xmax:4d}, {ymax:4d}] | Size: {width:3d}x{height:3d}")
    
    # Detectar sobreposições
    for j in range(i):
        other = objects[j]
        other_bbox = other.find('bndbox')
        ox_min = int(other_bbox.find('xmin').text)
        oy_min = int(other_bbox.find('ymin').text)
        ox_max = int(other_bbox.find('xmax').text)
        oy_max = int(other_bbox.find('ymax').text)
        
        # Verificar se há interseção
        if not (xmax < ox_min or ox_max < xmin or ymax < oy_min or oy_max < ymin):
            overlaps.append((i, j, name, other.find('name').text))

print("\n" + "=" * 100)
if overlaps:
    print(f"⚠️  SOBREPOSIÇÕES ENCONTRADAS ({len(overlaps)}):")
    for i, j, name1, name2 in overlaps[:10]:
        print(f"   • Objeto {i+1} ({name1}) sobrepõe Objeto {j+1} ({name2})")
    if len(overlaps) > 10:
        print(f"   ... e {len(overlaps)-10} mais")
else:
    print("✅ Nenhuma sobreposição detectada")

print("\n" + "=" * 100)
print("📈 ESTATÍSTICAS:")
sizes = [int(o.find('bndbox').find('xmax').text) - int(o.find('bndbox').find('xmin').text) for o in objects]
print(f"   Largura média das boxes: {sum(sizes)/len(sizes):.0f}px")
print(f"   Menor box: {min(sizes)}px")
print(f"   Maior box: {max(sizes)}px")
