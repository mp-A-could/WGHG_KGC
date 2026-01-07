# WGHG_KGC
Knowledge Graph Completion Based on Wavelet Graph Attention and Hyperbolic-Space Gating

A knowledge graph completion model combining Wavelet-enhanced Graph Attention Networks with Hyperbolic space prediction.

## Architecture

- **WaveletRGATEncoder**: Encodes entity and relation embeddings using Chebyshev wavelet transforms and relational graph attention
- **HyperbolicKGPred**: Predicts tail entities in hyperbolic (Lorentz) space with semantic gating and Givens rotations
- **BERT Text Encoder**: Extracts semantic features from entity names and descriptions

## Project Structure

```
├── main.py                          # Entry point and training script
├── helper.py                        # Utility functions
├── kgc_data.py                      # Data loading and preprocessing
├── models/
│   ├── P_model_standalone.py        # Main model (KGCPromptTuner) and Trainer
│   ├── WaveletRGATEncoder.py        # Wavelet + GAT encoder
│   ├── HyperbolicKGPred.py          # Hyperbolic prediction module
│   └── neighbor_semantic_sim.py     # Neighbor semantic similarity computation
├── data/processed/                  # Processed datasets
└── checkpoint/                      # Model checkpoints
```

## Requirements

- torch >= 2.7.1
- torch-geometric >= 2.6.1
- torch-scatter >= 2.1.2
- torch-sparse >= 0.6.18
- transformers >= 4.53.0
- tqdm >= 4.67.1
- numpy
- pandas

## Usage

### Training

```bash
python main.py -dataset WN18RR -batch_size 200 -epoch 100
```

### Testing

```bash
python main.py -dataset WN18RR -model_path checkpoint/WN18RR/WN18RR-best.pth
```

### Continue Training

```bash
python main.py -dataset WN18RR -continue_path checkpoint/WN18RR/WN18RR-best.pth
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-dataset` | WN18RR | Dataset name |
| `-batch_size` | 200 | Training batch size |
| `-epoch` | 100 | Number of training epochs |
| `-embed_dim` | 128 | Entity embedding dimension |
| `-gat_heads` | 4 | Number of GAT attention heads |
| `-cheb_K` | 4 | Chebyshev polynomial order |
| `-neg_K` | 300 | Number of negative samples |
| `-text_score_weight` | 7 | Text score weight for inference |
| `-lr` | 0.001 | Learning rate |
