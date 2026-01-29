# 📋 Ordem de Execução

## Passo 1: Preparar Dataset (se não feito)
```bash
python cloud_architecture_detector.py
```
Converte PNG+XML para formato YOLO.

## Passo 2: Verificar Ambiente
```bash
python quick_test.py
```
Valida se tudo está pronto.

## Passo 3: Treinar Modelo (Terminal 1)
```bash
python start_training.py
```
Treina YOLOv8 + RAG (4-12 horas).

## Passo 4: Monitorar GPU (Terminal 2)
```bash
python monitor_gpu.py
```
Acompanha utilização da GPU em tempo real.

## Passo 5: Fazer Predições (após treino)
```bash
python inference.py
```
Testa modelo em novas imagens.

## Passo 6: Visualizar (opcional)
```bash
python visualization.py
```
Desenha bounding boxes e mostra resultados.

## Passo 7: Deploy API (opcional)
```bash
python api_server.py
```
Inicia servidor REST na porta 5000.

---

**Comece por:** `python quick_test.py`
