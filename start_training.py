#!/usr/bin/env python3
"""
Start Training - Script simplificado para treinar apenas YOLO (sem RAG)
Detecta bounding boxes em imagens de arquitetura em nuvem
"""

from pathlib import Path
import sys
import shutil
import json
from ultralytics import YOLO
import yaml


def main():
    print("\n" + "=" * 70)
    print("🚀 INICIANDO TREINAMENTO YOLO (DETECÇÃO DE OBJETOS)")
    print("=" * 70 + "\n")
    epoch = 100

    try:
        # Configurar caminhos
        dataset_dir = Path("./dataset_prepared")
        classes_file = Path("./dataset_prepared/classes.txt")

        # 1. Verificar dataset
        print("1️⃣  Verificando dataset...")
        if not dataset_dir.exists():
            print("   ❌ Execute primeiro: python cloud_architecture_detector.py")
            return False

        labels_dir = dataset_dir / "labels"
        if not labels_dir.exists():
            print("   ❌ Pasta de labels não encontrada")
            return False

        images_dir = dataset_dir / "images"
        if not images_dir.exists():
            print("   ❌ Pasta de imagens não encontrada")
            return False

        num_images = len(list(images_dir.glob("*.png"))) + len(
            list(images_dir.glob("*.jpg"))
        )
        num_labels = len(list(labels_dir.glob("*.txt")))
        print(f"   ✅ Dataset OK: {num_images} imagens, {num_labels} anotações")

        # 2. Criar data.yaml
        print("\n2️⃣  Criando configuração YOLO (data.yaml)...")

        # Ler classes (formato: "ID NOME_DA_CLASSE")
        with open(classes_file) as f:
            classes = []
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    # Pegar apenas o nome da classe (segunda parte)
                    classes.append(parts[1])
                elif parts:
                    classes.append(parts[0])

        # Dividir dataset em train/val
        all_images = sorted(
            list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
        )
        split_idx = int(len(all_images) * 0.8)

        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]

        # Criar arquivo data.yaml com lista simples de nomes
        data_yaml = {
            "path": str(dataset_dir.absolute()),
            "train": "images",
            "val": "images",
            "nc": len(classes),
            "names": classes,  # Lista simples de nomes
        }

        yaml_path = Path("./data.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"   ✅ data.yaml criado com {len(classes)} classes")
        print(f"      • Train: {len(train_images)} imagens")
        print(f"      • Val: {len(val_images)} imagens")

        # 3. Treinar modelo YOLO
        print("\n3️⃣  Iniciando treinamento YOLO...")
        print("   ⏱️  Tempo estimado: 30 min - 2 horas (depende de GPU)")
        print("   💡 Abra outro terminal e execute: python monitor_gpu.py\n")

        model = YOLO("yolov8m.pt")  # YOLOv8 Medium - bom balanço precisão/velocidade

        results = model.train(
            data=str(yaml_path),
            epochs=epoch,
            imgsz=640,
            device=0,  # GPU 0
            patience=5,
            save=True,
            verbose=True,
        )

        print("\n   ✅ Treinamento concluído!")

        # 4. Copiar modelo para cloud_detector_model
        print("\n4️⃣  Salvando modelo em cloud_detector_model/...")

        model_dir = Path("./cloud_detector_model")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Copiar pesos YOLO
        best_weights_src = Path("./runs/detect/train/weights/best.pt")
        best_weights_dst = model_dir / "detection_model.pt"

        if best_weights_src.exists():
            shutil.copy2(best_weights_src, best_weights_dst)
            print(
                f"   ✅ Pesos copiados: detection_model.pt ({best_weights_dst.stat().st_size / 1e6:.1f} MB)"
            )
        else:
            print(f"   ⚠️  Arquivo não encontrado: {best_weights_src}")

        # Criar metadata
        metadata = {
            "model_size": "m",
            "device": "cuda",
            "provider": None,
            "epochs": epoch,
            "classes": len(classes),
        }
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ Metadata criado: metadata.json")

        # 5. Resumo
        print("\n" + "=" * 70)
        print("✅ TREINAMENTO COMPLETO COM SUCESSO!")
        print("=" * 70)
        print("\n📊 Arquivos gerados:")
        print(f"   📁 {model_dir.absolute()}/")
        for f in sorted(model_dir.glob("*")):
            size = f.stat().st_size / 1e6
            print(f"      └─ {f.name} ({size:.1f} MB)")

        print("\n📊 Próximos passos:")
        print("   • Teste em imagens: python inference.py")
        print("   • Visualize resultados: python visualization.py")

        return True

    except KeyboardInterrupt:
        print("\n\n⏸️  Treinamento interrompido pelo usuário")
        return False
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
