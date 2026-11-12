# Tunisian Speech AI: TEDxTN

Reproducible project scaffold for Tunisian Arabic ASR and Tunisian Arabic-to-English speech translation using the public TEDxTN corpus.

## Status and scope

- The public metadata was inspected from Hugging Face dataset revision `2f25fbea14b584a181f01f00aa4c6acbe352a6f7` (repository snapshot checked 29 October 2026); the inspection script pins this revision when downloading metadata.
- The Hugging Face repository contains README files, `TEDxTN_URLS.txt`, and six CSV files; its API reports `modality:text` and zero stored dataset bytes. No WAV, MP3, FLAC, or other audio asset is stored there. `TEDxTN_URLS.txt` contains external YouTube video URLs and talk IDs; the paper also points readers to those external videos. These URLs are references, not permission to download, train on, or redistribute the recordings. Obtain authorization and follow the source-platform terms before using audio.
- A local audio audit is implemented and has run. It found 0 of 108 expected WAV files in the workspace; all expected filenames are recorded in `results/metrics/audio_inventory.json`. No ASR or translation model was trained or evaluated.
- The metadata and EDA notebook cells were executed successfully. The active Python environment has CPU-only `torch 2.6.0+cpu`; `torch.cuda.is_available()` is false, and `transformers`, `datasets`, and `sacrebleu` are not installed. Model-inference and training cells remain unexecuted.
- The notebooks are designed to report missing audio and keep model runs opt-in. Results are not fabricated; local experiment result files are only written after an experiment runs.

## Findings from the actual metadata

| Split | Tunisian source rows | English translation rows | Talks |
|---|---:|---:|---:|
| train | 15,703 | 15,701 | 97 |
| dev / validation | 731 | 731 | 5 |
| test | 842 | 842 | 6 |
| **Total** | **17,276** | **17,274** | **108** |

The talk identifiers do not overlap across the three source splits. There is one whitespace-only source transcript in the inspected CSV snapshot; blank English references occur 9/1/1 times for train/dev/test. Four train segments are longer than Whisper's 30-second input limit. The English CSV boundaries are not always identical to source boundaries, and one training talk has a different segment count (249 source rows vs 247 English rows). The project therefore reports ordinal/time-overlap candidates (15,454 train, 731 dev, 842 test before blank-target filtering), requiring at least 50% coverage of both intervals, and excludes the unmatched talk instead of assuming exact CSV row joins. The measured minimum source-interval overlap among candidates is 99.82%. Candidate alignment still needs manual spot checks.

The ASR metadata EDA measured 30,309 distinct whitespace-delimited train tokens. A script-based count found 163,463 Arabic-containing and 41,846 Latin-containing token instances, with zero containing both scripts. This is a script heuristic, not a language or code-switch annotation.

The CSV columns are:

- Tunisian source: `audio_filename`, `start_time`, `end_time`, `transcription`
- English target: `audio_filename`, `start_time`, `end_time`, `translation`

The files contain no speaker IDs, region labels, or dialect-variant annotations. Audio sample rates cannot be measured from the hosted metadata. The paper reports WAV audio sampled at 16 kHz. The current CSV row totals also differ from the paper's Table 5 counts, so the project treats the downloaded metadata snapshot as authoritative for its own split reports and does not present the two versions as identical.

## Paper and experimental rationale

Bougares et al. introduce 108 Tunisian TEDx talks (28h39 collected audio; 25h01 transcribed), with code-switching across Tunisian Arabic, French, and English. Their guidelines use Arabic script for Tunisian words, Roman script for foreign words, and dialect-aware fixed spellings; translations prioritize natural meaning. They split by complete talks and report no speaker overlap between train, validation, and test.

The paper fine-tunes Whisper, MMS, XLS-R, and w2v-BERT-2.0 using SpeechBrain for 80 epochs. Its best reported ASR is w2v-BERT-2.0T (21.37% test WER, 8.34% CER); its Whisper large-v3 ASR test WER is 25.19%. Whisper large-v3 direct speech translation reaches 25.68 test BLEU. These are paper results, not results produced by this repository. The paper reports that zero-shot Whisper-large-v3 WER remains high (94.00% test), illustrating the value of dialect-specific adaptation. The reported limitations include limited geographic and demographic coverage, plus the corpus's non-commercial/no-derivatives license.

The selected practical reproduction path is Whisper-small ASR fine-tuning and Whisper-small direct speech translation, with Whisper `translate` as the direct speech baseline and NLLB-200 distilled 600M (`aeb_Arab` → `eng_Latn`) for the ASR→MT pipeline. The pinned Whisper model card documents both transcription and translation task tokens and lists Arabic; the pinned NLLB card lists Tunisian Arabic and English. Those labels do not establish benchmark quality. NLLB's model card describes local Transformers inference and marks hosted inference unavailable. The experiment config pins model revisions. Whisper-small is Apache-2.0; NLLB-200 distilled 600M is CC-BY-NC-4.0 and should be treated as research-only/non-commercial. The local RTX 3050 has 4 GiB VRAM and driver 511.65 (CUDA 11.6 reported), so its memory and driver may not support current CUDA wheels; full fine-tuning is intended for a suitable Colab or larger GPU. Smoke-test settings are provided.

## Layout

```text
.
├── README.md
├── requirements.txt
├── configs/experiment_config.json
├── notebooks/
│   ├── 01_Tunisian_ASR.ipynb
│   └── 02_Speech_to_Text_Translation.ipynb
├── scripts/
│   ├── inspect_dataset.py
│   └── check_audio.py
├── src/
│   ├── data_utils.py
│   └── evaluation.py
├── tests/test_project.py
├── report/research_report.md
├── results/
│   ├── metrics/
│   │   ├── dataset_inspection.json
│   │   ├── audio_inventory.json
│   │   └── audio_inventory_test.json
│   └── figures/
│       └── asr_metadata_eda.png
└── .gitignore
```

Optional text-normalization and dialect-identification notebooks are not included: the mandatory speech experiments need to be operational first, and the CSVs do not provide ground-truth dialect labels.

## Setup

Use Python 3.10–3.12. Install a PyTorch build compatible with your runtime first, then install the remaining requirements. In Colab, keep its preinstalled PyTorch unless a verified compatibility issue requires otherwise:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements.txt
```

`torch` is intentionally not pinned in `requirements.txt` because the correct wheel depends on platform and CUDA runtime. For GPU training, select a wheel compatible with the installed NVIDIA driver. The notebook checks `torch.cuda.is_available()` and otherwise uses CPU; CPU training is not recommended for a full run. In this workspace, installed PyTorch is CPU-only (`2.6.0+cpu`) even though an RTX 3050 is present; Transformers, Datasets, and sacreBLEU are not installed.

## Inspect the dataset

Download only the small public metadata files (no talk audio):

```powershell
python scripts\inspect_dataset.py --download-metadata
```

The command writes `results/metrics/dataset_inspection.json`. For an already downloaded directory:

```powershell
python scripts\inspect_dataset.py --dataset-dir C:\path\to\tedxtn-metadata
```

Run the lightweight utility tests:

```powershell
python -m unittest discover -s tests -v
```

## Audio preparation

The loader expects exactly one WAV per talk named `data/audio/<audio_filename>.wav` by default. It decodes the `[start_time, end_time)` interval from that talk file, converts to mono, and resamples to 16 kHz. Set `TEDXTN_AUDIO_DIR` to use another directory; it must contain these same filenames (subfolders and MP3/FLAC files are not automatically searched or converted). The paper describes 16 kHz WAV, but the loader also handles other sample rates.

No audio downloader is included. Obtain and prepare recordings only if you are authorized; do not commit audio. To see the exact missing files, audit the user-authorized directory:

```powershell
$env:TEDXTN_AUDIO_DIR = 'D:\authorized-data\TEDxTN\audio'
python scripts\check_audio.py
```

The audit writes `results/metrics/audio_inventory.json`, checks WAV headers, sample rates, segment bounds, and decodes one metadata-selected segment per available talk. It exits nonzero while any required WAV is missing or invalid; this is an intentional readiness signal. It currently lists all 108 talk filenames as missing. The ASR baseline needs all six test-talk files. Fine-tuning and its held-out test evaluation need all 108 talks across train/dev/test. Translation needs the six test-talk files.

## Running experiments

Install dependencies in a compatible Python environment, set `TEDXTN_AUDIO_DIR`, and open each notebook from the project root (or its `notebooks` folder). Run cells top-to-bottom. Metadata loading, text EDA, alignment candidates, and missing-audio reporting work without audio. In `01_Tunisian_ASR.ipynb`, set `RUN_MODEL_EVALUATION=True` to run the eight-example smoke baseline; set `RUN_TRAINING=True` as well to smoke-test adaptation on at most 16 train and 8 dev segments. Keep `RUN_SMOKE_TEST=True` for that check. For a full run, set `RUN_SMOKE_TEST=False`; first run and inspect the baseline, then enable training. Model runs remain disabled by default.

In `02_Speech_to_Text_Translation.ipynb`, inspect the displayed alignment examples spanning all six test talks. Set `ALIGNMENT_REVIEWED=True` only after review, then set `RUN_TRANSLATION=True` for the eight-example smoke comparison. For full scoring, set `RUN_SMOKE_TEST=False`. This script deliberately does not train a translation model: Whisper direct translation is zero-shot, and the second system is zero-shot ASR followed by NLLB MT.

ASR scoring reports WER/CER under two explicit text policies: minimally normalized NFC/whitespace and punctuation/case/whitespace normalization. WER uses whitespace-delimited tokens; CER removes whitespace before character-level scoring. Neither policy rewrites Tunisian words into Modern Standard Arabic. Translation is evaluated with default sacreBLEU tokenization and chrF; the sacreBLEU signature is saved. The paper reports true-cased BLEU without punctuation, so project scores should not be compared directly unless scoring is made identical. The same held-out test candidates are used by both translation approaches. The test split is not used for training or model selection.

Model checkpoints, caches, audio, and other large artifacts are ignored by Git. Machine-readable experiment metrics and qualitative examples should be saved under `results/` after running the notebooks; no performance value should be entered unless measured.

## License and citations

TEDxTN metadata is published under **CC BY-NC-ND 4.0**. Review the license and source terms before using or distributing audio, adapted artifacts, or model checkpoints. The audio itself is not part of the Hugging Face repository.

- Bougares, Fethi, Salima Mdhaffar, Haroun Elleuch, and Yannick Estève. 2025. “TEDxTN: A Three-way Speech Translation Corpus for Code-Switched Tunisian Arabic - English.” *Proceedings of the Third Arabic Natural Language Processing Conference*, 278–287. https://doi.org/10.18653/v1/2025.arabicnlp-main.22
- Dataset: https://huggingface.co/datasets/fbougares/TEDxTN
- Whisper: https://huggingface.co/openai/whisper-small
- NLLB-200: https://huggingface.co/facebook/nllb-200-distilled-600M
