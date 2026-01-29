#!/usr/bin/env python3
"""
Quick Test - Verifica se tudo está pronto para treinar
"""

from pathlib import Path
import sys

def check_setup():
    """Valida a configuração"""
    
    print("\n" + "="*70)
    print("✅ VERIFICAÇÃO DE PRÉ-REQUISITOS")
    print("="*70 + "\n")
    
    errors = []
    warnings = []
    
    # 1. Verificar dataset
    print("📂 Checando dataset...")
    dataset_dir = Path('./dataset')
    if dataset_dir.exists():
        pngs = list(dataset_dir.glob('*.png'))
        xmls = list(dataset_dir.glob('*.xml'))
        print(f"   ✅ Dataset encontrado")
        print(f"      • {len(pngs)} imagens PNG")
        print(f"      • {len(xmls)} anotações XML")
        
        if len(pngs) == 0 or len(xmls) == 0:
            errors.append("Dataset vazio!")
    else:
        errors.append("Dataset não encontrado em ./dataset")
    
    # 2. Verificar dataset_prepared
    print("\n📁 Checando dataset_prepared...")
    prepared_dir = Path('./dataset_prepared')
    if prepared_dir.exists():
        labels_dir = prepared_dir / 'labels'
        classes_file = prepared_dir / 'classes.txt'
        
        if labels_dir.exists():
            txt_files = list(labels_dir.glob('*.txt'))
            print(f"   ✅ Pasta labels encontrada")
            print(f"      • {len(txt_files)} arquivos de anotação YOLO")
            
            if len(txt_files) == 0:
                warnings.append("Nenhum arquivo de anotação YOLO encontrado")
        else:
            errors.append("Pasta 'labels' não encontrada em dataset_prepared")
        
        if classes_file.exists():
            with open(classes_file) as f:
                classes = f.readlines()
            print(f"   ✅ Arquivo classes.txt encontrado")
            print(f"      • {len(classes)} classes")
        else:
            errors.append("Arquivo 'classes.txt' não encontrado")
    else:
        errors.append("Pasta 'dataset_prepared' não encontrada")
    
    # 3. Verificar dependências
    print("\n📦 Checando dependências...")
    deps = {
        'torch': 'PyTorch',
        'ultralytics': 'YOLOv8',
        'transformers': 'Transformers (CLIP)',
        'PIL': 'Pillow (Image)',
    }
    
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            errors.append(f"Faltando: {name} ({module})")
    
    # 4. Verificar GPU
    print("\n🎮 Checando GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA disponível")
            print(f"      • GPU: {torch.cuda.get_device_name(0)}")
            print(f"      • Memória: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            warnings.append("CUDA não disponível, usará CPU (muito mais lento)")
    except Exception as e:
        errors.append(f"Erro ao verificar GPU: {e}")
    
    # Resumo
    print("\n" + "="*70)
    print("📊 RESUMO")
    print("="*70)
    
    if errors:
        print("\n❌ ERROS ENCONTRADOS:")
        for i, err in enumerate(errors, 1):
            print(f"   {i}. {err}")
    
    if warnings:
        print("\n⚠️  AVISOS:")
        for i, warn in enumerate(warnings, 1):
            print(f"   {i}. {warn}")
    
    if not errors:
        print("\n✅ TUDO PRONTO PARA TREINAR!")
        print("\n   Próximos passos:")
        print("   1. Execute: python training_pipeline.py")
        print("   2. Em outro terminal: python monitor_gpu.py")
        return True
    else:
        print("\n❌ CONFIGURE OS ERROS ACIMA ANTES DE CONTINUAR")
        return False

if __name__ == '__main__':
    success = check_setup()
    sys.exit(0 if success else 1)
