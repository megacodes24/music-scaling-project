# music-scaling-project

This project explores scaling laws for language models trained on symbolic music data, comparing Transformer and LSTM architectures on ABC notation from the Lakh MIDI dataset.

## Overview

I investigate how model performance scales with parameter count by:
- Training 5 Transformer models
- Training 4 LSTM models with comparable parameter counts
- Fitting power law curves: L = a·N^(-α) + c
- Analyzing computational efficiency and musical quality

## Project Structure
```
│   ├── raw_midi/           # Downloaded MIDI files
│   ├── abc_notation/       # Converted ABC files
│   ├── train.txt           # Training data (98%)
│   ├── val.txt             # Validation data (1%)
│   ├── test.txt            # Test data (1%)
│   ├── tokenizer.pkl       # Character-level tokenizer
│   └── failed_midi.txt     # Conversion failures log
│
├── checkpoints/
│   ├── transformer_*.pt    # Trained transformer models
│   ├── lstm_*.pt           # Trained LSTM models
│   └── best_model.pt       # Best model checkpoint
│
├── part4_outputs/
│   ├── generated/          # Generated ABC samples
│   │   ├── sample_*.abc
│   │   └── metadata.json
│   ├── checkpoints/
│   │   ├── best_model.pt
│   │   └── training_history.json
│   └── RESULTS.json        # Final evaluation results
└── requirements.txt

```

### 2. Data Preparation

**Download the Lakh MIDI Dataset:**
# It downloads the clean_midi subset (~17K files)
import urllib.request
url = "http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz"
urllib.request.urlretrieve(url, "clean_midi.tar.gz")

**Convert MIDI to ABC notation:**


# Run the MIDI to ABC conversion
# This uses music21 library and creates ABC files in data/abc_notation/
# The script handles ~1265 files and reports success rate


**Build vocabulary and create splits:**

# The tokenizer builds character-level vocabulary
# Data is split: 98% train / 1% val / 1% test
# Sequences are filtered: 100 < length < 5000 characters


### 3. Model Training

**Train Transformer scaling suite :**


# Trains 5 transformer sizes for 1 epoch each
# Configurations: tiny , small , medium, large, xl
# Results saved to checkpoints/transformer_*.pt


**Train LSTM scaling suite :**


# Trains 4 LSTM sizes for 1 epoch each
# Results saved to checkpoints/lstm_*.pt


**Train best model:**

# Trains the XL transformer for 10 epochs
# Saves best checkpoint based on validation loss
# Generates 10 music samples with different prompts/temperatures


## Model Architectures

### Transformer Configurations

| Size   | Parameters| d_model | n_heads | n_layers | d_ff  |
|--------|-----------|---------|---------|----------|-------|
| Tiny   | ~0.14M    | 64      | 2       | 2        | 256   |
| Small  | ~0.67M    | 128     | 2       | 3        | 512   |
| Medium | ~1.9M     | 192     | 4       | 4        | 768   |
| Large  | ~4.1M     | 256     | 4       | 5        | 1024  |
| XL     | ~7.6M     | 320     | 4       | 6        | 1280  |

### LSTM Configurations

| Size   | Parameters| Embedding | Hidden | Layers |
|--------|-----------|-----------|--------|--------|
| Tiny   | ~0.24M    | 64        | 128    | 2      |
| Small  | ~0.95M    | 128       | 256    | 2      |
| Medium | ~3.2M     | 192       | 384    | 3      |
| Large  | ~5.8M     | 256       | 512    | 3      |


## Dataset Statistics


- **Vocabulary size:** ~75 characters
- **Token counts:**
  - Training: ~2.3M tokens
  - Validation: ~28K tokens
  - Test: ~18K tokens
- **Sequence lengths:**
  - Mean: ~3014 characters


## Results Summary

### Scaling Law Findings

**Transformer:**
- Scaling exponent α: ~0.1484
- R² fit: ~0.9943

**LSTM:**
- Scaling exponent α: ~0.7904
- R² fit: ~0.9274

### Best Model Performance

- **Test Perplexity:** 14.61
- **ABC Validity Rate:** 100.0%


## Limitations

1. **Computational constraints:** Training on CPU is very slow; GPU recommended
2. **Dataset size:** Only used ~1265 files due to very slow conversion/filtering
3. **ABC quality:** Some generated samples may not be syntactically perfect

## Future Work

- Train on full Lakh dataset (176K files)
- Implement music-aware tokenization (note-level, event-based)
- Add attention visualization for interpretability
- Compare with other architectures (Mamba, etc.)
- Fine-tune on specific genres or composers

Key references:
- Kaplan et al. (2020): "Scaling Laws for Neural Language Models"
- Lakh MIDI Dataset: https://colinraffel.com/projects/lmd/
- ABC Notation Standard: https://abcnotation.com/wiki/abc:standard
