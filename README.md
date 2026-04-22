# DAiSEE Multimodal Engagement Classifier

A deep learning project for detecting and classifying student engagement levels using multimodal data (video scenes, facial expressions, and audio) from the DAiSEE dataset.

## 📋 Project Overview

This project implements a state-of-the-art multimodal engagement classifier that processes:
- **Face crops** - Facial expressions and head movements
- **Scene frames** - Body posture and classroom context
- **Audio** - Speech patterns and tone

The model combines these modalities using attention-based fusion to classify engagement into 4 levels:
- **0**: No Engagement
- **1**: Low Engagement
- **2**: Medium Engagement
- **3**: High Engagement

## Architecture

### Components

1. **Encoders**
   - `EfficientNetEncoder`: Extracts spatial features from face and scene frames
   - `AudioCNNEncoder`: Processes mel-spectrograms from video audio
   
2. **Temporal Modeling**
   - `TransformerTemporalEncoder`: Multi-head attention-based temporal fusion
   - `BiLSTMTemporalEncoder`: Alternative bidirectional LSTM approach

3. **Fusion**
   - `AttentionFusion`: Learns optimal weighting of face, scene, and audio embeddings

4. **Classification Head**
   - LayerNorm → Linear (256→128) → GELU → Dropout → Linear (128→4)

## Key Features

- **Disk-Caching System**: Preprocessed clips cached as `.npz` files for efficient loading
- **Parallel Processing**: Multi-threaded caching with `ThreadPoolExecutor`
- **Mixed Precision**: CUDA AMP support for faster training
- **Modular Design**: Easy to swap encoders, temporal models, and fusion strategies
- **FFmpeg Integration**: Robust audio extraction with error handling
- **Progressive Logging**: Real-time training metrics and ETA calculation

## Quick Start

### Requirements
```bash
pip install torch torchvision torchaudio
pip install facenet-pytorch librosa tqdm
pip install pandas opencv-python
```

### Configuration

Edit the `Config` class in the notebook to customize:
```python
cfg.BATCH_SIZE  = 8          # Batch size for training
cfg.EPOCHS      = 20         # Number of training epochs
cfg.LR          = 1e-4       # Learning rate
cfg.MAX_FRAMES  = 16         # Frames per video clip
cfg.EMBED_DIM   = 256        # Embedding dimension
cfg.DEVICE      = "cuda"     # "cuda" or "cpu"
```

### Training

```python
model, optimizer, scheduler, best_val_acc = train()
```

The training pipeline:
1. **Builds label dictionaries** from CSV files
2. **Creates disk cache** from raw video files
3. **Initializes dataloaders** with multi-worker support
4. **Trains model** with validation monitoring
5. **Saves best checkpoint** based on validation accuracy

### Evaluation

```python
evaluate_test()
```

Evaluates the best model on test set and reports loss and accuracy.

## 📊 Dataset Structure

```
DAiSEE/
├── DataSet/
│   ├── Train/
│   ├── Validation/
│   └── Test/
└── Labels/
    ├── TrainLabels.csv
    ├── ValidationLabels.csv
    └── TestLabels.csv
```

Each label CSV contains: `ClipID`, `Engagement` (0-3), and other optional fields.

## 🔧 Implementation Details

### Video Processing
- Face detection: MTCNN with margin=20
- Face crops: 112×112 (float32)
- Scene crops: 224×224 with ImageNet normalization
- Frame sampling: Adaptive based on video FPS
- Padding: Dummy frames added to reach `MAX_FRAMES`

### Audio Processing
- Extraction: FFmpeg (PCM 16-bit, mono, 16kHz)
- Representation: Mel-spectrogram (64 mels, hop_length=512)
- Normalization: Per-clip min-max scaling
- Storage: float16 for efficiency

### Training Strategy
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: Cosine Annealing (T_max=20 epochs)
- **Loss**: Cross-Entropy
- **Gradient Clipping**: max_norm=1.0
- **Validation**: Best model saved by validation accuracy

## Output Files

After training, the following files are saved:

- `best_daisee_model.pt` - Model weights only
- `full_checkpoint.pt` - Complete training state + config
- `config.json` - Hyperparameters in JSON format

## 🎯 Model Specifications

| Parameter | Value |
|-----------|-------|
| Max Frames | 16 |
| Embedding Dim | 256 |
| Transformer Heads | 8 |
| Transformer Layers | 2 |
| Dropout | 0.3 |
| Batch Size | 8 |
| Total Trainable Params | ~88M+ |

## 💾 Memory & Performance

- **GPU Memory**: ~8-12GB (with batch_size=8)
- **Cache Size**: ~2-5GB for full dataset
- **Training Time**: ~2-4 hours per epoch (single GPU)
- **Inference Speed**: ~50-100ms per clip

## 🔍 Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE`
- Reduce `EMBED_DIM`
- Enable gradient checkpointing

### Missing Cache Files
- Ensure all video files exist and are readable
- Check FFmpeg installation for audio extraction
- Verify video codecs are supported by OpenCV

### Audio Extraction Fails
- Install FFmpeg: `apt-get install ffmpeg` (Linux) or download from ffmpeg.org
- Check video file integrity

## 📝 Notebooks

- **CV-1(0.5).ipynb** - Complete training pipeline with all components
- **CV-2(0.6).ipynb** - Alternative approaches or extended analysis

## 🤝 Contributing

Modifications and extensions welcome:
- Add new encoder architectures
- Experiment with different temporal models
- Implement advanced augmentation strategies
- Extend to other engagement-related tasks

## 📚 References

- DAiSEE Dataset: [Kaggle Dataset](https://www.kaggle.com/datasets/olgaparfenova/daisee)
- EfficientNet: [Paper](https://arxiv.org/abs/1905.11946)
- MTCNN: [Paper](https://arxiv.org/abs/1604.02878)
- Transformer: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

**Last Updated**: April 2026
