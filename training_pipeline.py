"""
Cloud Architecture Detector - Training Pipeline
Pipeline de treinamento para detecção de arquitetura cloud
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json

try:
    from ultralytics import YOLO
    import torch
    import torchvision
except ImportError:
    print("⚠️  Dependências não encontradas. Execute: pip install -r requirements.txt")


class CloudArchitectureDetector:
    """
    Detector de arquitetura cloud usando YOLO
    """

    def __init__(self, model_size: str = "m", device: str = None, provider: str = None):
        """
        Inicializa o detector

        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
            device: 'cuda' ou 'cpu' (auto-detecta se None)
            provider: 'aws', 'azure', 'gcp' ou None (para modelo multi-provider)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"🚀 Usando device: {self.device}")

        # Inicializar YOLO para detecção de objetos
        self.detection_model = YOLO(f"yolov8{model_size}.pt")
        self.detection_model.to(self.device)

        self.model_size = model_size
        self.provider = provider

    def _get_optimal_batch_size(self):
        """Detecta batch size ideal baseado em GPU"""
        if not torch.cuda.is_available():
            print("  ℹ️  GPU não disponível, usando CPU (batch_size=8)")
            return 8

        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  ℹ️  GPU memory: {gpu_memory:.1f}GB")

            if gpu_memory >= 16:
                batch_size = 32
            elif gpu_memory >= 8:
                batch_size = 16
            elif gpu_memory >= 4:
                batch_size = 8
            else:
                batch_size = 4

            print(f"  ✅ Batch size selecionado: {batch_size}")
            return batch_size

        except Exception as e:
            print(f"  ⚠️  Erro ao detectar GPU: {e}")
            return 8

    def prepare_dataset_structure(self, dataset_dir: Path, train_split: float = 0.8):
        """
        Prepara estrutura de dados no formato YOLO

        Args:
            dataset_dir: Diretório com imagens e labels
            train_split: Proporção de dados para treino

        Returns:
            Tupla com (imagens_treino, imagens_val)
        """
        import random
        from shutil import copy2

        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"

        if not images_dir.exists():
            raise ValueError(f"Diretório de imagens não encontrado: {images_dir}")
        if not labels_dir.exists():
            raise ValueError(f"Diretório de labels não encontrado: {labels_dir}")

        # Coletar todas as imagens
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            image_files.extend(list(images_dir.glob(ext)))

        if len(image_files) == 0:
            raise ValueError(f"Nenhuma imagem encontrada em {images_dir}")

        print(f"📊 Total de imagens encontradas: {len(image_files)}")

        # Verificar se labels existem para cada imagem
        valid_images = []
        for img in image_files:
            label_file = labels_dir / f"{img.stem}.txt"
            if label_file.exists():
                valid_images.append(img)
            else:
                print(f"⚠️  Label não encontrado para: {img.name}")

        if len(valid_images) == 0:
            raise ValueError("Nenhuma imagem com label correspondente encontrada!")

        print(f"✅ Imagens com labels: {len(valid_images)}")

        # Split train/val
        random.shuffle(valid_images)
        split_idx = int(len(valid_images) * train_split)

        train_images = valid_images[:split_idx]
        val_images = valid_images[split_idx:]

        print(f"📊 Split: {len(train_images)} treino / {len(val_images)} validação")

        # Criar estrutura YOLO
        yolo_dir = Path("yolo_dataset")
        yolo_dir.mkdir(exist_ok=True)

        for split, images in [("train", train_images), ("val", val_images)]:
            split_img_dir = yolo_dir / split / "images"
            split_lbl_dir = yolo_dir / split / "labels"
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_lbl_dir.mkdir(parents=True, exist_ok=True)

            for img in images:
                copy2(img, split_img_dir / img.name)
                label_file = labels_dir / f"{img.stem}.txt"
                if label_file.exists():
                    copy2(label_file, split_lbl_dir / f"{img.stem}.txt")

        return train_images, val_images

    def create_yolo_yaml(self, dataset_dir: Path, classes_file: Path):
        """
        Cria arquivo YAML de configuração para YOLO

        Args:
            dataset_dir: Diretório raiz do dataset YOLO
            classes_file: Arquivo com classes

        Returns:
            Path do arquivo YAML criado
        """
        # Ler classes
        with open(classes_file) as f:
            classes = []
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    classes.append(parts[1])

        yaml_content = f"""path: {dataset_dir.absolute()}
train: train/images
val: val/images

nc: {len(classes)}
names: {classes}
"""

        yaml_path = Path("data.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        print(f"✅ YAML criado: {yaml_path}")
        print(f"   Classes: {len(classes)}")

        return yaml_path

    def train_detection_model(
        self, yaml_path: Path, epochs: int = 100, imgsz: int = 640
    ):
        """
        Treina modelo de detecção YOLO

        Args:
            yaml_path: Caminho para configuração YAML
            epochs: Número de épocas
            imgsz: Tamanho da imagem

        Returns:
            Resultados do treinamento
        """
        print(f"\n{'='*70}")
        print(f"🎯 INICIANDO TREINAMENTO - YOLOv8{self.model_size.upper()}")
        print(f"{'='*70}")
        print(f"  📊 Épocas: {epochs}")
        print(f"  📐 Tamanho imagem: {imgsz}px")
        print(f"  🎮 Device: {self.device}")

        batch_size = self._get_optimal_batch_size()

        # Configurações otimizadas
        train_args = {
            "data": str(yaml_path),
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch_size,
            "device": self.device,
            "patience": 50,
            "save": True,
            "project": "runs/detect",
            "name": "train",
            "exist_ok": True,
            "pretrained": True,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "workers": 8,
            "amp": True,
            "plots": True,
            "deterministic": True,
            "verbose": True,
        }

        print(f"\n{'='*70}")
        print("🚀 Iniciando treinamento...")
        print(f"{'='*70}\n")

        results = self.detection_model.train(**train_args)

        print(f"\n{'='*70}")
        print("   ✅ Treinamento concluído!")
        print(f"{'='*70}\n")

        return results

    def predict(self, image_path: Path, confidence: float = 0.5) -> Dict:
        """
        Realiza detecção em uma imagem

        Args:
            image_path: Caminho da imagem
            confidence: Confiança mínima

        Returns:
            Dicionário com detecções
        """
        results = self.detection_model.predict(
            source=str(image_path), conf=confidence, device=self.device, verbose=False
        )

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

        return {"image_path": str(image_path), "detections": detections}

    def save_model(self, model_path: Path):
        """Salva modelo treinado"""
        model_path.mkdir(parents=True, exist_ok=True)

        # Copiar pesos do YOLO
        best_weights = Path("runs/detect/train/weights/best.pt")
        if best_weights.exists():
            import shutil

            shutil.copy2(best_weights, model_path / "detection_model.pt")

        # Salvar metadata
        metadata = {
            "model_size": self.model_size,
            "device": self.device,
            "provider": self.provider,
        }

        with open(model_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Modelo salvo em: {model_path}")


def main():
    """Exemplo de uso"""
    detector = CloudArchitectureDetector(model_size="m")

    # Preparar dataset
    dataset_dir = Path("dataset_prepared")
    classes_file = dataset_dir / "classes.txt"

    train_imgs, val_imgs = detector.prepare_dataset_structure(dataset_dir)
    yaml_path = detector.create_yolo_yaml(Path("yolo_dataset"), classes_file)

    # Treinar
    detector.train_detection_model(yaml_path, epochs=100)

    # Salvar
    detector.save_model(Path("cloud_detector_model"))


if __name__ == "__main__":
    main()
