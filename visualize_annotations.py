#!/usr/bin/env python3
"""
Visualiza as anotações XML sobre as imagens PNG
"""

from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import random

def visualize_annotations(sample_size=5):
    """Cria visualizações das anotações"""
    
    dataset_dir = Path('./dataset')
    output_dir = Path('./visualizations')
    output_dir.mkdir(exist_ok=True)
    
    pngs = sorted(dataset_dir.glob('*.png'))
    selected = random.sample(pngs, min(sample_size, len(pngs)))
    
    print(f"\n📊 Gerando {len(selected)} visualizações...\n")
    
    for png_path in selected:
        xml_path = png_path.with_suffix('.xml')
        
        if not xml_path.exists():
            print(f"⚠️  {png_path.name} -> XML não encontrado, pulando")
            continue
        
        # Carregar imagem
        img = Image.open(png_path)
        draw = ImageDraw.Draw(img)
        
        # Carregar anotações
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = root.findall('object')
        
        # Cores aleatórias para cada classe
        colors = {}
        
        # Desenhar bounding boxes
        for obj in objects:
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Assign color per class
            if name not in colors:
                colors[name] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
            color = colors[name]
            
            # Draw rectangle
            draw.rectangle(
                [(xmin, ymin), (xmax, ymax)],
                outline=color,
                width=3
            )
            
            # Draw label
            draw.text(
                (xmin, ymin - 10),
                name,
                fill=color
            )
        
        # Salvar
        output_path = output_dir / f"annotated_{png_path.name}"
        img.save(output_path)
        
        print(f"✅ {png_path.name}")
        print(f"   {len(objects)} objetos encontrados")
        print(f"   Salvo em: visualizations/annotated_{png_path.name}\n")
    
    print(f"\n✨ Visualizações salvas em: visualizations/")
    print("   Abra as imagens para verificar se as anotações estão corretas!")

if __name__ == '__main__':
    visualize_annotations(sample_size=5)
