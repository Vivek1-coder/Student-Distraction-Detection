# DAiSEE Multimodal Engagement Classifier

A deep learning project for detecting and classifying student engagement levels using multimodal data (video scenes, facial expressions, and audio) from the DAiSEE dataset.

## Project Overview

This project implements a multimodal engagement classifier that combines video, facial, and audio data to understand student engagement. It processes three key data streams:
- **Face crops** — Facial expressions and head movements
- **Scene frames** — Body posture and overall classroom context
- **Audio** — Speech patterns and tone of voice

The model fuses these three data sources together using attention mechanisms to classify engagement into one of four levels:
- **0** — No engagement
- **1** — Low engagement
- **2** — Medium engagement
- **3** — High engagement

## Architecture

The model consists of several interconnected components:

### Encoders
- **EfficientNetEncoder** — Extracts spatial features from face and scene frames
- **AudioCNNEncoder** — Processes mel-spectrograms from the video audio
### Temporal Modeling
- **TransformerTemporalEncoder** — Uses multi-head attention to model temporal patterns across frames
- **BiLSTMTemporalEncoder** — Alternative approach using bidirectional LSTM layers

### Fusion Layer
- **AttentionFusion** — Learns how to optimally weight and combine embeddings from all three modalities

### Classification Head
The final layers take the fused representation and classify it: LayerNorm → Linear (256→128) → GELU → Dropout → Linear (128→4)

## Key Features

- **Smart Caching** — Preprocessed video clips are stored as `.npz` files, so they only need to be processed once
- **Parallel Processing** — Uses multiple threads to cache videos quickly without blocking training
- **Mixed Precision Training** — CUDA AMP automatically reduces precision where possible for faster, more efficient training
- **Modular Components** — Easy to swap out encoders, temporal models, or fusion strategies for experimentation
- **Robust Audio Handling** — Integrated FFmpeg extraction with fallbacks for problematic files
- **Real-time Monitoring** — Logs training progress with live metrics and time estimates

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

Start training with:
```python
model, optimizer, scheduler, best_val_acc = train()
```

Here's what happens during training:
1. Load engagement labels from the CSV files
2. Preprocess all videos and cache them locally
3. Set up data loaders with parallel workers
4. Train the model while monitoring validation performance
5. Save the best model based on validation accuracy

### Evaluation

```python
evaluate_test()
```

Evaluates the best model on test set and reports loss and accuracy.

## Dataset Structure

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

## Implementation Details

### Video Processing
- **Face detection** — MTCNN with a 20-pixel margin around detected faces
- **Face crops** — Resized to 112×112 pixels at full precision (float32)
- **Scene crops** — Full frames resized to 224×224 with ImageNet normalization
- **Frame sampling** — Adaptive: we extract fewer frames from slow videos and more from fast ones
- **Padding** — Videos with fewer than MAX_FRAMES are padded with blank frames to standardize input

### Audio Processing
- **Extraction** — FFmpeg extracts mono audio at 16kHz in 16-bit PCM format
- **Representation** — Converted to mel-spectrograms with 64 frequency bins
- **Normalization** — Each audio clip is normalized independently to the [0,1] range
- **Storage** — Stored at reduced precision (float16) to save disk space

### Training Strategy
- **Optimizer** — AdamW with a learning rate of 1e-4 and weight decay of 1e-4
- **Learning Rate Schedule** — Cosine annealing that smoothly reduces the learning rate over 20 epochs
- **Loss Function** — Cross-entropy for multi-class classification
- **Gradient Clipping** — Limits gradient magnitude to 1.0 to prevent instability
- **Model Selection** — Saves the best model based on validation accuracy

## Output Files

After training finishes, three files are saved:

- **best_daisee_model.pt** — Just the model weights, good for inference
- **full_checkpoint.pt** — Everything: model weights, optimizer state, scheduler state, and config
- **config.json** — All hyperparameters in a human-readable format

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Frames per clip | 16 |
| Embedding dimension | 256 |
| Attention heads | 8 |
| Transformer layers | 2 |
| Dropout rate | 0.3 |
| Batch size | 8 |
| Trainable parameters | ~88M+ |

## Memory & Performance

- **GPU Memory** — Around 8–12GB needed (with batch_size=8)
- **Disk Cache** — Full dataset takes roughly 2–5GB of storage
- **Training Speed** — Approximately 2–4 hours per epoch on a single GPU
- **Inference** — Around 50–100ms to process a single video clip

## Troubleshooting

### Running out of GPU memory?
- Try reducing `BATCH_SIZE` to 4 or lower
- Reduce `EMBED_DIM` from 256 to 128
- Enable gradient checkpointing for memory efficiency

### Some videos aren't being cached?
- Make sure all video files exist and are readable
- Verify FFmpeg is installed correctly for audio extraction
- Check that your video files use codecs that OpenCV supports

### Audio extraction fails?
- Install FFmpeg: run `apt-get install ffmpeg` on Linux, or download it from ffmpeg.org
- Test if the video files are corrupted or damaged

## Notebooks

- **CV-1(0.5).ipynb** — The main notebook with the complete training pipeline
- **CV-2(0.6).ipynb** — Explores alternative architectures and deeper analysis (takes 24–36 hours)

## References

- DAiSEE Dataset: [Kaggle Dataset](https://www.kaggle.com/datasets/olgaparfenova/daisee)
- EfficientNet: [Paper](https://arxiv.org/abs/1905.11946)
- MTCNN: [Paper](https://arxiv.org/abs/1604.02878)
- Transformer: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---
**Last Updated**: April 2026
