# Arquitetura - Cloud Architecture Detector com RAG

## 🏗️ Visão Geral da Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    CLOUD ARCHITECTURE DETECTOR              │
│                     (RAG + Object Detection)                │
└─────────────────────────────────────────────────────────────┘

                           INPUT
                             │
              ┌──────────────┴──────────────┐
              │                             │
         ┌────▼────┐              ┌────────▼──────┐
         │  Imagem │              │  XML Annotation│
         │  (PNG)  │              │   (PASCAL VOC) │
         └────┬────┘              └────────┬──────┘
              │                             │
              │                    ┌────────▼──────────┐
              │                    │ XML Parser        │
              │                    │ - Ler classes     │
              │                    │ - Extrair bboxes  │
              │                    └────────┬──────────┘
              │                             │
         ┌────┴─────────────────────────────┴────┐
         │                                       │
    ┌────▼────────────────┐        ┌────────────▼──────┐
    │ CloudArchitectureDS  │        │ YOLOv8 Conversion │
    │ - Parser XML        │        │ - Normalizar bbox │
    │ - Class mapping     │        │ - Gerar .txt      │
    │ - Statistics        │        └────────┬──────────┘
    └────┬────────────────┘                  │
         │                                    │
         │      TRAINING DATA PREPARED
         │                                    │
    ┌────┴────────────────────────────────────┴────┐
    │                                              │
┌───▼──────────────────┐          ┌───────────────▼──┐
│ CloudArchitectureRAG │          │ Data Splitter    │
│ Detector             │          │ (80/20 train/val)│
└───┬──────────────────┘          └───────────────┬──┘
    │                                             │
    ├─────────────────────┬───────────────────────┤
    │                     │                       │
┌───▼──────────────┐  ┌──▼──────────────┐   ┌──▼──────────────┐
│ YOLO Detection   │  │ CLIP Model      │   │ Data Preparation
│ Model            │  │ (RAG Embeddings)│   │ (YOLO format)
│                  │  │                 │   └──┬───────────────┘
│ - Training       │  │ - Extract image │      │
│ - Validation     │  │   features      │      │
│ - Inference      │  │ - Build index   │  ┌───▼────────────────┐
│                  │  │ - Similarity    │  │ Metrics & Analysis
└───┬──────────────┘  └──┬──────────────┘  │ - Accuracy
    │                    │                 │ - Precision/Recall
    │                    │                 │ - mAP Score
    │                 ┌──┴────────────┐    └──────┬──────────────┘
    │                 │              │           │
    │          ┌──────▼──────┐   ┌───▼──────┐   │
    │          │ Embeddings  │   │ Inference│   │
    │          │ Index (CLIP)│   │ Pipeline │   │
    │          └──┬───┬──────┘   └───┬──────┘   │
    │             │   │              │          │
    │             │   │    ┌─────────┴───────┐  │
    │             │   │    │   Single Image  │  │
    │             │   │    │   Prediction    │  │
    │             │   │    └────┬──────────┬─┘  │
    │             │   │         │          │    │
    │             │   │         │    ┌─────▼────┴──┐
    │             │   │         │    │              │
    │             │   └────┬────┴────┤ Batch        │
    │             │        │        │ Processing   │
    │             │    ┌───▼────────▼─────┐        │
    │             │    │ RAG Retrieval    │        │
    │             └────┤ - Vector search  │        │
    │                  │ - Top-K similar  │        │
    │                  │ - Context use    │        │
    │                  └───┬──────────────┘        │
    │                      │                       │
    │                 ┌────▼───────────┐           │
    │                 │ Visualization  │           │
    │                 │ - Draw bboxes  │◄──────────┘
    │                 │ - Plot metrics │
    │                 │ - Report gen   │
    │                 └────┬───────────┘
    │                      │
    └──────────┬───────────┘
               │
          ┌────▼────────────────────┐
          │    SAVED MODEL OUTPUT   │
          ├────────────────────────┤
          │ - detection_model.pt   │
          │ - rag_embeddings.pt    │
          │ - classes.txt          │
          │ - data.yaml            │
          └────────────────────────┘
               │
          ┌────▼────────────────────┐
          │  INFERENCE / DEPLOYMENT │
          ├────────────────────────┤
          │ API REST               │
          │ Batch processing       │
          │ Real-time detection    │
          └────────────────────────┘
```

## 📦 Componentes Principais

### 1. **Data Layer** (`cloud_architecture_detector.py`)

```python
CloudArchitectureDataset
├── parse_xml_annotation()      # Ler anotações PASCAL VOC
├── load_dataset()              # Carregar todas as imagens
├── create_yolo_format_annotations()  # Converter para YOLO
├── create_class_mapping_file()       # Salvar classes
└── get_statistics()                  # Gerar estatísticas
```

**Transformações**:
- PASCAL VOC XML → YOLO format (.txt files)
- Bounding boxes normalizados (0-1)
- Class mapping (string → int ID)

### 2. **Model Training Layer** (`training_pipeline.py`)

```python
CloudArchitectureRAGDetector
├── __init__()                  # Carregar YOLOv8 + CLIP
├── prepare_dataset_structure() # Split train/val
├── create_yolo_yaml()         # Criar config YAML
├── train_detection_model()    # YOLOv8 training
├── build_rag_index()          # Gerar embeddings
├── retrieve_similar_images()  # RAG search
├── predict_with_rag()         # Inferência completa
└── save_model()               # Persistir modelo
```

**Arquitetura Dual**:
- **YOLOv8**: Object Detection (Fast, Accurate)
- **CLIP**: Vision-Language Model (Semantic Understanding)

### 3. **Inference Layer** (`inference.py`)

```python
CloudArchitectureInference
├── __init__()                 # Load saved model
├── detect_and_retrieve()      # Single image inference
├── process_batch()            # Batch processing
└── export_results()           # JSON export
```

### 4. **Visualization Layer** (`visualization.py`)

```python
DetectionVisualizer
├── draw_detections()          # Draw bboxes on image
├── plot_detection_results()   # Matplotlib visualization
├── create_class_distribution_chart()  # Statistics chart
└── _get_color()              # Colorize by class

AnalysisReport
├── generate_summary()         # Statistical summary
└── print_report()            # Formatted output
```

## 🔄 Fluxo de Dados

### Fase 1: Preparação

```
Dataset (6000 PNG + 6000 XML)
    │
    ├─► XML Parser
    │   └─► Bounding boxes, Classes
    │
    └─► Image Loader
        └─► PIL/OpenCV

    ↓

YOLO Format (.txt)
├─ class_id x_center y_center width height
└─ (Normalizado 0-1)

    ↓

Split Train/Val (80/20)
├─ train/images/
├─ train/labels/
├─ val/images/
└─ val/labels/
```

### Fase 2: Treinamento

```
YOLOv8 Path:
    Input → Backbone → Neck → Head → Output
    (Image) (Feature extraction) (Predictions)

Training Loop:
    ├─ Forward pass
    ├─ Loss computation
    ├─ Backward pass
    ├─ Gradient update
    ├─ Validation
    └─ Checkpoint (repeat)

CLIP Path (Parallel):
    Input → Vision Transformer → Image Embeddings
    (Image) (Feature extraction) (Vector representation)

    ├─ For each training image
    ├─ Generate embedding (512-dim vector)
    └─ Store in index (FAISS or numpy)
```

### Fase 3: Inferência

```
Query Image
    │
    ├─► YOLOv8 Detection Pipeline
    │   ├─ Backbone extraction
    │   ├─ Neck processing
    │   ├─ Head predictions
    │   └─ Post-processing (NMS)
    │       └─ Detections: [class, bbox, conf]
    │
    ├─► CLIP Embedding
    │   ├─ Image preprocessing
    │   ├─ Vision encoder
    │   └─ Feature vector (512-dim)
    │
    └─► RAG Retrieval
        ├─ Vector similarity (cosine)
        ├─ Index search
        └─ Top-K results
            └─ Similar images paths

Output:
{
    "image_path": str,
    "detections": [
        {"class_id": int, "class_name": str, "confidence": float, "bbox": [x1,y1,x2,y2]},
        ...
    ],
    "similar_images": [
        {"path": str, "similarity": float},
        ...
    ]
}
```

## 🧠 Modelos Utilizados

### YOLOv8 (Object Detection)

```
Arquitetura:
┌──────────────────────────────────────────┐
│ Input (640x640)                          │
└────────┬─────────────────────────────────┘
         │
    ┌────▼──────────────────────────┐
    │ Backbone (CSPDarknet)          │
    │ - Conv layers com skip conn.   │
    │ - Feature pyramid              │
    │ - Resolution: 640→320→160→80   │
    └────┬──────────────────────────┘
         │
    ┌────▼──────────────────────────┐
    │ Neck (PANet)                   │
    │ - Feature aggregation          │
    │ - Multi-scale fusion           │
    └────┬──────────────────────────┘
         │
    ┌────▼──────────────────────────┐
    │ Head                           │
    │ - Class prediction             │
    │ - Bbox regression              │
    │ - Confidence scoring           │
    └────┬──────────────────────────┘
         │
    ┌────▼──────────────────────────┐
    │ Output                         │
    │ - Detections (x1,y1,x2,y2)    │
    │ - Class logits                 │
    │ - Confidence scores            │
    └────────────────────────────────┘
```

**Tamanhos**:
- Nano (n):   3.2M params
- Small (s):  11.2M params
- Medium (m): 25.9M params
- Large (l):  43.7M params

### CLIP (Vision-Language Model)

```
Arquitetura:
┌──────────────────────────────────────────┐
│ Image Input (224x224)                    │
└────────┬─────────────────────────────────┘
         │
    ┌────▼──────────────────────────┐
    │ Vision Transformer (ViT)       │
    │ - Patch embedding (16x16)      │
    │ - Transformer encoder (12 layers) │
    │ - Output: [batch, 512]         │
    └────┬──────────────────────────┘
         │
    ┌────▼──────────────────────────┐
    │ Image Embeddings (512-dim)     │
    │ - Normalized L2 vectors        │
    │ - Suitable for similarity      │
    └────────────────────────────────┘

RAG Index:
    ├─ Image 1: [0.23, -0.15, ..., 0.45]
    ├─ Image 2: [0.24, -0.14, ..., 0.46]
    ├─ Image 3: [0.10,  0.20, ..., 0.30]
    └─ ... (6000 embeddings)

Similarity Search:
    Query embedding × Index embeddings = Scores
    Top-5 índices com scores mais altos
```

## 💾 Estrutura de Arquivos de Saída

### Modelo Salvo

```
cloud_detector_model/
├── detection_model.pt         # YOLOv8 treinado (PT file)
│   └─ Pesos + arquitetura
│
├── rag_embeddings.pt          # CLIP embeddings index
│   └─ {path_str: np.array([512])}
│
└── classes.txt                # Mapeamento de classes
    └─ 0 aws_amazon_api_gateway
       1 aws_lambda_lambda_function
       ...
```

### Resultado de Predição (JSON)

```json
{
  "image_path": "/path/to/image.png",
  "detections": [
    {
      "class_id": 0,
      "class_name": "aws_amazon_api_gateway",
      "confidence": 0.9478,
      "bbox": [641.2, 480.5, 747.8, 579.3]
    },
    {
      "class_id": 5,
      "class_name": "aws_lambda_lambda_function",
      "confidence": 0.8765,
      "bbox": [1353.1, 1778.4, 1428.9, 1849.2]
    }
  ],
  "similar_images": [
    {
      "path": "/path/to/train_image_0234.png",
      "similarity": 0.9412
    },
    {
      "path": "/path/to/train_image_0567.png",
      "similarity": 0.9087
    }
  ]
}
```

## 🔌 Integrações Possíveis

### REST API (Flask/FastAPI)

```python
@app.post("/detect")
def detect_image(file: UploadFile):
    image = Image.open(file.file)
    result = inference.detect_and_retrieve(image_path)
    return result

@app.post("/batch")
def batch_detection(directory: str):
    results = inference.process_batch(directory)
    return results
```

### Real-time Streaming (OpenCV)

```python
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    results = detector.predict_with_rag(frame)
    visualized = visualizer.draw_detections(frame, results)
    cv2.imshow('Detection', visualized)
```

### Database Integration

```python
# Salvar embeddings em vector DB
db = Milvus()
for img_path, embedding in detector.image_embeddings.items():
    db.insert({
        "image_path": img_path,
        "embedding": embedding.tolist()
    })
```

## 📊 Métricas de Performance

### Para YOLOv8

```
mAP@50    - Mean Average Precision (50% IoU)
mAP@75    - Mean Average Precision (75% IoU)
Precision - TP / (TP + FP)
Recall    - TP / (TP + FN)
F1-score  - Harmônico entre Precision e Recall
```

### Para RAG

```
Embedding Quality:
  - Cosine similarity (0-1)
  - Euclidean distance
  - Recall@K (retrieval accuracy)

Inference Speed:
  - Detection latency (ms)
  - Embedding time (ms)
  - Total end-to-end time
```

## 🚀 Próximas Melhorias

1. **Data Augmentation**
   - Random rotation, flip, brightness
   - Mosaic augmentation
   - Mixup/Cutmix

2. **Multi-scale Testing**
   - Test time augmentation
   - Ensemble predictions

3. **Quantization**
   - INT8 quantization (4x faster)
   - ONNX export

4. **Ensemble Methods**
   - Multiple model sizes
   - Voting/averaging predictions

5. **Active Learning**
   - Hard example mining
   - Uncertainty sampling

---

**Diagrama Mantido e Atualizado**
