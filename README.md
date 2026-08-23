# MediAssist

Generates a radiology report from a chest X-ray. An image encoder reads the
scan, an attention layer picks out the regions that matter, and a Transformer
decoder writes the findings text.

Built for the GenAI course project, trained on MIMIC-CXR.

## Model

Three parts.

**Image encoder.** ResNet-50, outputs a 2048 x 7 x 7 feature map for each
image. Frozen for the first 5 epochs so the decoder can settle before the
encoder starts moving.

**Spatial attention.** Turns the 49 spatial positions into a saliency map so
the decoder can weight regions rather than treating the image as one flat
vector.

**Text decoder.** 2-layer Transformer, d_model 512, 4 heads, feedforward 2048,
dropout 0.1. Cross-attends the text being generated against the visual
features. Tokenizer is `bert-base-uncased`, sequences capped at 128 tokens.

## Training

| Setting | Value |
|---|---|
| Epochs | 100 |
| Batch size | 8 |
| Learning rate (decoder) | 1e-4 |
| Learning rate (encoder) | 1e-5 |
| Encoder freeze | first 5 epochs |
| Gradient clipping | 1.0 |
| Image size | 224 x 224 |
| Mixed precision | on |
| Gradient checkpointing | on |

The encoder gets a learning rate 10x smaller than the decoder. It comes in
pretrained, so large updates early on would wreck features that are already
good. Mixed precision and gradient checkpointing are both there to fit the
model in available memory - batch size 8 is what that leaves room for.

## Results

Beam search at width 5 on the held-out test split:

| Metric | Score |
|---|---|
| BLEU-1 | 0.3101 |
| BLEU-2 | 0.2156 |
| BLEU-3 | 0.1566 |
| BLEU-4 | 0.1174 |
| ROUGE-1 | 0.3377 |
| ROUGE-2 | 0.1404 |
| ROUGE-L | 0.2672 |

BLEU falling off from 0.31 at unigrams to 0.12 at 4-grams is the usual shape
for report generation - the model gets the clinical vocabulary right more
often than it gets whole phrases right.

## Running it

Set your paths and hyperparameters in `config.py` first, then:

```
python mediassist_train.py              # train
python mediassist_evaluate.py           # predictions on the test split
python mediassist_calculate_metrics.py  # BLEU / ROUGE
python mediassist_visualize.py          # training curves and metric plots
python mediassist_predict.py --image <path>   # single image
```

`setup_environment.bat` sets up the environment on Windows, or install from
`requirements.txt` directly.

## Layout

| Path | Contents |
|---|---|
| `config.py` | All paths and hyperparameters |
| `mediassist_model.py` | Encoder, attention, decoder |
| `mediassist_dataset.py` | Loading and preprocessing |
| `mediassist_train.py` | Training loop |
| `mediassist_evaluate.py` | Test set inference |
| `mediassist_calculate_metrics.py` | Scoring |
| `mediassist_visualize.py` | Plots |
| `mediassist_predict.py` | Single-image inference |
| `train_split.csv`, `valid_split.csv`, `model_test.csv` | Dataset splits |
| `checkpoints/` | Weights and training history |
| `results/` | Predictions and `metrics.json` |
| `report.pdf` | Write-up |
