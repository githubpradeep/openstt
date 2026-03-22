# Fast LJSpeech STT

Minimal, from-scratch speech-to-text built for the fastest useful training loop, not the best benchmark score.

This project uses:

- LJSpeech
- 80-bin log-mel spectrograms
- character-level targets
- a small CNN frontend
- a small BiLSTM encoder
- greedy CTC decoding
- PyTorch and torchaudio

It does not use pretrained encoders, transformers, RNNT, Whisper, wav2vec2, raw-waveform models, or language-model fusion.

## Why this setup

### Why LJSpeech

LJSpeech is a good fastest-training choice because it is small, clean, single-speaker, and easy to load from a standard folder layout. You can get a working STT pipeline without spending time on multi-speaker cleanup, noisy alignments, or heavy storage requirements.

### Why log-mel instead of raw waveform

Log-mel features remove a large amount of low-level signal complexity before the model ever sees the input. That makes the model smaller, easier to debug, and much faster to train than raw-waveform approaches.

### Why CNN + BiLSTM + CTC

This is a strong simple baseline for end-to-end speech recognition:

- CNN layers extract local patterns and reduce time resolution cheaply.
- BiLSTM layers model left and right context without an attention decoder.
- CTC keeps the training loop simple and avoids alignment labels.

For this project, that tradeoff is better than a larger transformer or a more complicated sequence criterion.

### Why character-level targets

Character vocabularies are tiny, deterministic, and easy to inspect. There is no separate lexicon, subword training, or language model. That makes debugging much simpler for a first working baseline.

## Project tree

```text
stt/
├── README.md
├── audio.py
├── config.py
├── dataset.py
├── infer.py
├── metrics.py
├── model.py
├── prepare_data.py
├── requirements.txt
├── stt_quickstart.ipynb
├── text.py
├── train.py
├── utils.py
└── configs
    ├── ljspeech_fast.yaml
    └── tiny_debug.yaml
```

## Download LJSpeech from Hugging Face

This repo now includes [prepare_data.py](/Users/pradeep.borado/misc/stt/prepare_data.py), which downloads the official Hugging Face `keithito/lj_speech` dataset and writes it into the standard LJSpeech folder layout expected by training.

Download the full dataset:

```bash
python prepare_data.py --output-dir data/LJSpeech-1.1
```

Optional partial download for quick setup checks:

```bash
python prepare_data.py --output-dir data/LJSpeech-1.1 --max-samples 200
```

Rebuild the destination folder from scratch:

```bash
python prepare_data.py --output-dir data/LJSpeech-1.1 --force
```

## LJSpeech layout

Expected dataset structure:

```text
data/
└── LJSpeech-1.1/
    ├── metadata.csv
    ├── README
    └── wavs/
        ├── LJ001-0001.wav
        ├── LJ001-0002.wav
        └── ...
```

The training script reads the standard `metadata.csv` and creates JSONL manifests automatically in the run directory.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then download LJSpeech:

```bash
python prepare_data.py --output-dir data/LJSpeech-1.1
```

## Recommended config to start with

Start with `configs/tiny_debug.yaml`.

It is the fastest path to checking that:

- data loading works
- mel shapes are correct
- target lengths are correct
- CNN output lengths still satisfy CTC
- the model can overfit a tiny subset

After that, move to `configs/ljspeech_fast.yaml`.

## Training

### Tiny debug run

Overfits a tiny subset quickly.

```bash
python train.py --config configs/tiny_debug.yaml --dataset-root data/LJSpeech-1.1
```

### Train on 100 samples

```bash
python train.py --config configs/ljspeech_fast.yaml --dataset-root data/LJSpeech-1.1 --subset-size 100
```

### Train on 500 samples

```bash
python train.py --config configs/ljspeech_fast.yaml --dataset-root data/LJSpeech-1.1 --subset-size 500
```

### Train on 1000 samples

`configs/ljspeech_fast.yaml` already defaults to `subset_size: 1000`.

```bash
python train.py --config configs/ljspeech_fast.yaml --dataset-root data/LJSpeech-1.1
```

### Train on full LJSpeech

Set `dataset.subset_size: null` in the config, or make a copy of the config with:

```yaml
dataset:
  subset_size: null
```

Then run:

```bash
python train.py --config configs/ljspeech_fast.yaml --dataset-root data/LJSpeech-1.1
```

### Resume training

```bash
python train.py \
  --config configs/ljspeech_fast.yaml \
  --dataset-root data/LJSpeech-1.1 \
  --resume runs/ljspeech_fast/last.pt
```

## What training prints

During training and validation, the script prints:

- train loss
- val loss
- CER
- WER
- learning rate
- sample greedy predictions
- sanity-check tensor shapes before real training starts

## Inference

### One file

```bash
python infer.py --checkpoint runs/ljspeech_fast/best.pt --audio path/to/file.wav
```

### Folder

```bash
python infer.py \
  --checkpoint runs/ljspeech_fast/best.pt \
  --audio-dir path/to/wavs \
  --batch-size 8 \
  --output runs/ljspeech_fast/predictions.jsonl
```

## Config notes

Key fast-iteration settings:

- `dataset.subset_size`: `100`, `500`, `1000`, or `null`
- `dataset.seed`: deterministic split and subset selection
- `dataset.overfit_mode`: reuses the same tiny subset for train and val
- `audio.sample_rate`: defaults to `16000`
- `audio.n_mels`: defaults to `80`
- `model.cnn_channels`: small channels for speed
- `model.cnn_time_strides`: controls time downsampling
- `training.amp`: mixed precision on CUDA
- `training.batch_size`: increase until memory becomes the limit

## Common CTC mistakes

This repo explicitly handles the most common CTC failure modes:

- blank token is fixed to index `0`
- targets are concatenated instead of padded for `CTCLoss`
- audio is padded, but target sequences are tracked separately
- feature lengths are computed from waveform lengths
- output lengths are recomputed after CNN downsampling
- `log_softmax` is applied before `CTCLoss`
- empty normalized transcripts are filtered out during manifest generation
- greedy decoding collapses repeats and removes blanks correctly

If `output_lengths < target_lengths`, training will raise an error instead of silently producing bad CTC batches.

## Debugging bad runs

### If loss becomes NaN or inf

- lower the learning rate
- disable masking augmentations
- confirm `output_lengths >= target_lengths`
- confirm sample rate and hop length are what you expect
- confirm transcripts are not empty after normalization
- keep `zero_infinity=True` in `CTCLoss`

### If predictions stay blank

- first run `configs/tiny_debug.yaml`
- make sure the model can overfit the 20-sample overfit subset
- reduce CNN downsampling if output sequences are too short
- inspect the printed sample predictions before and after training

### If shapes look wrong

The sanity path in `train.py` prints:

- waveform batch shape
- mel batch shape
- target tensor shape
- output tensor shape
- min and max output lengths

## Expected limitations

This project is intentionally minimal, so its limitations are also intentional:

- LJSpeech is single-speaker, so generalization is limited
- character decoding is simple but not the most accurate
- greedy CTC decoding is fast but weaker than beam search
- the model is optimized for fast training and clarity, not state-of-the-art quality

## Files and outputs

Training creates:

- `runs/<experiment>/resolved_config.yaml`
- `runs/<experiment>/vocab.json`
- `runs/<experiment>/manifests/*.jsonl`
- `runs/<experiment>/last.pt`
- `runs/<experiment>/best.pt`
- optional `runs/<experiment>/epoch_XXX.pt`

## Notebook

`stt_quickstart.ipynb` mirrors the script workflow if you want a single notebook entrypoint for setup, training commands, and inference commands.
