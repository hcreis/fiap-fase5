#!/usr/bin/env python3
"""
Diagnóstico simples de anotações incompletas
"""

from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
import json

dataset_dir = Path('./dataset')
pngs = sorted(dataset_dir.glob('*.png'))

print("\n📊 DIAGNÓSTICO DE ANOTAÇÕES INCOMPLETAS\n")
print("-" * 90)

# Análise rápida
stats = {
    'total_imgs': len(pngs),
    'imgs_0_objetos': 0,
    'imgs_1_a_5_objetos': 0,
    'imgs_6_a_10_objetos': 0,
    'imgs_10plus_objetos': 0,
    'avg_objects': 0,
}

objeto_counts = []

for png in pngs[:50]:  # Analisar primeiras 50
    xml = png.with_suffix('.xml')
    try:
        tree = ET.parse(xml)
        root = tree.getroot()
        count = len(root.findall('object'))
        objeto_counts.append(count)
        
        if count == 0:
            stats['imgs_0_objetos'] += 1
        elif count <= 5:
            stats['imgs_1_a_5_objetos'] += 1
        elif count <= 10:
            stats['imgs_6_a_10_objetos'] += 1
        else:
            stats['imgs_10plus_objetos'] += 1
    except:
        pass

if objeto_counts:
    stats['avg_objects'] = sum(objeto_counts) / len(objeto_counts)

print(f"Análise das primeiras {len(objeto_counts)} imagens:")
print(f"  • Média de objetos por imagem: {stats['avg_objects']:.1f}")
print(f"  • Imagens sem objetos: {stats['imgs_0_objetos']}")
print(f"  • Imagens com 1-5 objetos: {stats['imgs_1_a_5_objetos']}")
print(f"  • Imagens com 6-10 objetos: {stats['imgs_6_a_10_objetos']}")
print(f"  • Imagens com 10+ objetos: {stats['imgs_10plus_objetos']}")

print("\n" + "=" * 90)
print("🔍 PROBLEMA IDENTIFICADO:")
print("""
   Você relatou que há ícones não marcados em ALGUMAS imagens.
   Isso significa:
   
   ❌ PROBLEMA: Anotações INCOMPLETAS
      - Nem todos os objetos foram anotados
      - O modelo vai aprender mal
      - Vai produzir previsões ruins
      
   💡 SOLUÇÕES:
   
   1️⃣  REANNOTAR MANUALMENTE (melhor, mas lento)
       → Usar LabelImg para corrigir as imagens problemáticas
       
   2️⃣  GERAR ANOTAÇÕES COM MODELO PRÉ-TREINADO (automático)
       → Usar YOLOv8 pré-treinado para sugerir boxes faltantes
       
   3️⃣  REMOVER IMAGENS RUINS (rápido)
       → Descartar imagens com cobertura < 80%
       
   4️⃣  TREINAR DE QUALQUER FORMA
       → Treinar mesmo sabendo que há lacunas
       → Modelo será inferior, mas pode funcionar
       
   ❓ QUAL VOCÊ PREFERE?
""")

