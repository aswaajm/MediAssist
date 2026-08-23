# MediAssist

Generates a radiology report from a chest X-ray. A CNN reads the scan, an
attention gate weights the regions that matter, and a Transformer decoder
writes the findings text one token at a time.

Built for the GenAI course project. Trained on chest radiographs from
MIMIC-CXR with their paired reports.

## Data

Three CSV splits, each with two columns - `image_path` and `report_text`.

| Split | Studies |
|---|---|
| `train_split.csv` | 7,000 |
| `valid_split.csv` | 1,000 |
| `model_test.csv` | 2,000 |

Images are loaded as RGB, resized to 224x224 and normalised with the usual
ImageNet statistics. Reports get wrapped as `[CLS] ... [SEP]`, tokenized with
`bert-base-uncased`, and padded or truncated to 128 tokens.

MIMIC-CXR has broken and missing image files scattered through it, so
`__getitem__` returns `None` on any load failure and a custom collate function
drops those before the batch is assembled. If an entire batch turns out to be
invalid it returns an empty batch that the training loop skips, rather than
crashing partway through an epoch.

## Model

`TienetReportGenerator` in `mediassist_model.py`, four pieces.

**Image encoder.** ResNet-50 pretrained on ImageNet with the classifier head
and average pool stripped off, leaving a 2048 x 7 x 7 feature map. A 1x1
convolution then reduces 2048 channels down to 512, so the visual features
live at the same width as the decoder and can be attended over directly. The
backbone is frozen for the first 5 epochs and unfrozen after, handled by
`set_epoch()` which the training loop calls each epoch.

**Spatial attention.** A 1x1 conv down to a single channel followed by a
sigmoid, producing one weight per spatial position. Features are multiplied by
that map, so the model can suppress irrelevant regions instead of treating the
image as one flat vector. The map is returned alongside the features, which
makes it available for visualising where the model looked.

**Positional encoding.** Standard sinusoidal, built once up to length 512.
Token embeddings are scaled by sqrt(d_model) before it is added.

**Text decoder.** Two decoder layers written by hand rather than using
`nn.TransformerDecoder`, because the layer here is pre-norm - LayerNorm runs
before each sub-block and the residual is added after, which trains more
stably at this depth than the post-norm arrangement PyTorch ships. Each layer
is self-attention over the text so far, cross-attention into the image
features, then a feedforward block. d_model 512, 4 heads, feedforward 2048,
dropout 0.1.

The 7x7 attended feature map is flattened into 49 tokens of width 512 and
handed to the decoder as cross-attention memory. Cross-attention weights come
back out of every layer, so per-token image attention is inspectable too.

A causal mask stops the decoder seeing future tokens, and a padding mask keeps
it off the padding.

## Training

Teacher forcing: the decoder is fed `input_ids[:, :-1]` and asked to predict
`input_ids[:, 1:]`. Loss is cross-entropy with `ignore_index` set to the pad
token, so padding contributes nothing to the gradient.

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate, decoder | 1e-4 |
| Learning rate, encoder | 1e-5 |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 2 |
| Epochs | 100 |
| Batch size | 8 |
| Encoder frozen | first 5 epochs |
| Gradient clipping | max norm 1.0 |
| Mixed precision | on, with GradScaler |

The encoder sits in its own parameter group at a learning rate ten times
smaller than everything else. It arrives pretrained while the decoder starts
from noise, so without that split the early gradients coming back through a
randomly-initialised decoder would wreck features that are already good. The
5-epoch freeze is the same idea taken further.

Batch size 8 is what fits once the ResNet and a 128-token sequence are both in
memory. Mixed precision is there for the same reason.

The loop is built to survive a long unattended run. A NaN loss skips the
update instead of poisoning the weights. A CUDA OOM empties the cache, steps
the scaler and moves to the next batch rather than dying. Every epoch that
improves validation loss overwrites the best checkpoint, so the saved weights
are the best ones seen and not simply the last ones.

### What the run actually did

Ran the full 100 epochs. Training loss fell from 10.40 to 1.05, validation
loss from 4.79 to 2.18.

Validation loss bottomed out at **2.1318 at epoch 25** and then drifted back
up and flattened - 2.172 at epoch 50, 2.172 at 75, 2.176 at 100 - while
training loss kept falling to 1.05. That widening gap is the model memorising
the training reports past epoch 25. Best-checkpoint saving means the weights
kept are the epoch-25 ones, so the scores below are unaffected, but roughly 75
epochs of that run were wasted compute.

`epochs_without_improvement` is tracked in the loop and printed, but nothing
breaks on it. Wiring it to a patience limit is the obvious fix and would have
cut the run by three quarters.

## Inference

Beam search, width 5, in `TienetReportGenerator.generate()`.

Sequences are scored by total log probability divided by `length ** 0.7`.
Without that penalty beam search reliably prefers short outputs, since every
extra token adds a negative log probability - the 0.7 exponent normalises for
length without over-rewarding rambling. Finished beams go into a completed
pool and are scored the same way, and identical candidate sequences are
filtered at each step so the beam does not fill up with duplicates.

Start and end tokens are resolved from whatever the tokenizer provides,
falling back through cls/bos and sep/eos to the pad token, so swapping
tokenizer does not break generation.

## Evaluation

`mediassist_evaluate.py` loads the best checkpoint, generates a report for
every row of the test split, and writes ground truth against generated to
`results/mediassist_test_predictions.csv`. Failures are recorded as
`GENERATION_FAILED` or `IMAGE_NOT_FOUND` rather than dropped, so the row count
stays honest.

`mediassist_calculate_metrics.py` scores that file. BLEU is corpus-level via
NLTK with Chen-Cherry smoothing over lowercased word-tokenized text. ROUGE is
F-measure with stemming, averaged per sample. Results land in
`results/metrics.json`.

| Metric | Score |
|---|---|
| BLEU-1 | 0.3101 |
| BLEU-2 | 0.2156 |
| BLEU-3 | 0.1567 |
| BLEU-4 | 0.1174 |
| ROUGE-1 | 0.3377 |
| ROUGE-2 | 0.1404 |
| ROUGE-L | 0.2673 |

BLEU dropping from 0.31 at unigrams to 0.12 at 4-grams is the usual shape for
report generation. The model picks up the clinical vocabulary well before it
gets whole phrasings right.

## Known limitations

- No early stopping. The counter exists but nothing acts on it, so the run
  went 75 epochs past its best validation loss.
- `TextEncoder` in `mediassist_model.py` is defined and never used - the
  decoder carries its own embedding. Dead code, safe to remove.
- BLEU and ROUGE reward n-gram overlap, not clinical correctness. A report can
  score well while getting the finding wrong. A real evaluation would need
  clinical-efficacy metrics over extracted findings.

## Running it

Set paths and hyperparameters in `config.py`, then:

```
python mediassist_train.py                     # train
python mediassist_evaluate.py                  # generate on the test split
python mediassist_calculate_metrics.py         # BLEU / ROUGE -> metrics.json
python mediassist_visualize.py                 # loss curves and metric plots
python mediassist_predict.py --image <path>    # single image
```

`setup_environment.bat` handles the environment on Windows, otherwise install
from `requirements.txt`. Needs torch, torchvision, transformers, pillow,
pandas, nltk, rouge-score, tqdm and numpy.

## Layout

| Path | Contents |
|---|---|
| `config.py` | Paths and every hyperparameter |
| `mediassist_model.py` | Encoder, spatial attention, decoder, beam search |
| `mediassist_dataset.py` | Dataset, transforms, filtering collate |
| `mediassist_train.py` | Training and validation loop |
| `mediassist_evaluate.py` | Test set generation |
| `mediassist_calculate_metrics.py` | BLEU / ROUGE scoring |
| `mediassist_visualize.py` | Loss curves, metric bar charts |
| `mediassist_predict.py` | Single-image inference |
| `train_split.csv`, `valid_split.csv`, `model_test.csv` | Splits |
| `checkpoints/` | Best weights, `training_history.json` |
| `results/` | Predictions CSV, `metrics.json` |
| `training_curves.png`, `metrics.png` | Generated plots |
| `report.pdf` | Write-up |
