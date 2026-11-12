# Tunisian Arabic Speech Recognition and Translation with TEDxTN

**Status:** Reproducible experiment scaffold; model experiments are not yet run.  
**Metadata snapshot inspected:** Hugging Face revision `2f25fbea14b584a181f01f00aa4c6acbe352a6f7` (checked 29 October 2026).

## Abstract

This project establishes a reproducible evaluation plan for Tunisian Arabic automatic speech recognition (ASR) and Tunisian Arabic-to-English speech translation using TEDxTN. The publicly hosted repository contains aligned task metadata and talk URLs, but does not host audio. The metadata inspection found 17,276 Tunisian transcript rows and 17,274 English translation rows across 108 talks, with no talk-ID overlap between train, validation, and test. Because audio has not been supplied, this report contains no new model scores. The experimental plan compares Whisper-small ASR fine-tuning with a zero-shot baseline, and compares a two-stage ASR→NLLB pipeline with direct Whisper speech translation.

## 1. Introduction

Tunisian Arabic is a low-resource, code-switched Arabic variety for which orthographic variation and spoken mixtures of Tunisian Arabic, French, and English complicate speech recognition and translation. This project asks:

1. How accurately does a multilingual pretrained ASR model transcribe Tunisian Arabic, and does task-specific adaptation improve it?
2. How does direct speech translation compare with an ASR followed by machine translation pipeline?

## 2. Related work

Bougares et al. (2025) introduce TEDxTN as a Tunisian Arabic-English speech-translation corpus. Their experiments fine-tune multilingual speech models and report a best test WER of 21.37% from w2v-BERT-2.0T ASR, and a best test BLEU of 25.68 from fine-tuned Whisper large-v3 speech translation. They find zero-shot Whisper transcription substantially worse than fine-tuned systems. Their work uses SpeechBrain and 80 training epochs.

## 3. Dataset and linguistic characteristics

### 3.1 Dataset documentation

The dataset card identifies Tunisian Arabic and CC BY-NC-ND 4.0. It describes 108 talks, about 25 hours of speech, code-switching, and speakers from more than 11 Tunisian regions. The paper states the collected source audio totals 28h39, of which 25h01 is transcribed, with 130 speakers in its overview table. Its narrative later mentions 133 speakers, an internal inconsistency.

The paper says audio was checked and stored as 16 kHz WAV and points users to YouTube talk URLs rather than bundling recordings. The Hugging Face API reports `modality:text`, zero stored bytes, and exactly README/URL files plus the six CSVs—no audio files in another format. The URL list is a documented external source reference, not permission to download, train on, or redistribute its recordings. The project does not fetch audio and requires user authorization and local files. The inspected source schema is `audio_filename`, `start_time`, `end_time`, `transcription`, and English target schema is `audio_filename`, `start_time`, `end_time`, `translation`.

### 3.2 Measured metadata snapshot

| Split | Source segments | English rows | Unique talks |
|---|---:|---:|---:|
| Train | 15,703 | 15,701 | 97 |
| Dev / validation | 731 | 731 | 5 |
| Test | 842 | 842 | 6 |
| **Total** | **17,276** | **17,274** | **108** |

The split talk sets are disjoint. The source CSV has a whitespace-only transcription; empty English references are also present. The English and Tunisian CSV boundaries are not exact-key matches, and one train talk has a different number of rows. The inspection helper only emits candidate bilingual pairings when the talk-level row counts agree and source/target intervals overlap; candidates need manual checks before scoring.

The current source CSV total differs from the paper's Table 3 (17,278 segments) and its Table 5 split total (15,626 + 731 + 842 = 17,199). This version discrepancy is retained as a data-integrity finding, not reconciled by silently dropping or adding rows. Conservative ordinal/time-overlap alignment produced 15,454 train, 731 dev, and 842 test candidates before excluding blank English references, requiring at least 50% coverage of each paired interval; the minimum measured source-interval coverage among candidates is 99.82%. The sole unmatched training talk has 249 source rows and 247 English rows. Manual spot checks remain required. Four training source segments exceed Whisper's 30-second input limit and are excluded by the ASR adaptation notebook. The paper reports no speaker overlap between splits; the public CSV schema inspected here has no speaker identifiers, so that claim cannot be independently rechecked from those CSV columns.

### 3.3 Linguistic properties and coverage

The paper describes non-standardized Tunisian spelling and code-switching, and guidelines that use Arabic script for Tunisian words, Latin script for unadapted foreign words, and natural rather than literal English translation. It reports 53.70% code-switched segments, a 21.50% corpus-level Code-Mixing Index, and 31,064 source vocabulary types. These are paper-reported statistics, not recomputed from the project metadata.

The public CSVs do not contain speaker IDs, region labels, or dialect-variant labels. Sample-rate distribution, corruption, and audio duration cannot be audited from the public CSVs alone. Audio-level EDA is deferred until authorized local WAVs are supplied.

## 4. Methodology

### 4.1 Metadata and audio preparation

Use the provided train/dev/test talk splits without resplitting. Keep each complete talk in its existing split. Validate intervals, transcript presence, duplicates, talk leakage, and English alignment. Exclude blank source transcripts and segments longer than 30 seconds from Whisper adaptation, recording the counts. Do not fetch or distribute audio automatically. When locally supplied, audio is expected at `data/audio/<audio_filename>.wav`; intervals are cropped, converted to mono, and resampled to 16 kHz for Whisper.

### 4.2 ASR

- Baseline: zero-shot `openai/whisper-small`, Arabic transcription task.
- Adapted model: fine-tune the same checkpoint on TEDxTN train, validate on dev, and evaluate once on test.
- Primary measures: corpus WER and CER with both NFC/whitespace-only and punctuation/case/whitespace normalization. WER uses whitespace-delimited tokens; CER removes whitespace before character-level edit distance. Neither normalizer converts dialectal text to MSA.
- Smoke tests use a small bounded number of segments and one epoch; they are implementation checks, not research results.

Whisper supports Arabic, but that does not establish robust Tunisian dialect or code-switch recognition. The small model is chosen for adaptation feasibility, not because it is known to outperform the paper's larger systems.

### 4.3 Speech translation

Use English as the target language because TEDxTN publishes English translations.

- Two-stage: ASR output from the selected ASR checkpoint → `facebook/nllb-200-distilled-600M` text translation with `aeb_Arab` source and `eng_Latn` target codes.
- Direct: zero-shot Whisper-small speech translation (`task="translate"`). The Whisper model card explicitly documents the multilingual speech-translation task token and recommends `get_decoder_prompt_ids(language=..., task="translate")`; this notebook passes language Arabic and the `translate` task. The base checkpoint supports Arabic and translation but is not Tunisian-fine-tuned. Direct speech-translation fine-tuning is not implemented in the current notebook.
- Measures: sacreBLEU and chrF with recorded sacreBLEU configuration. This project uses default sacreBLEU tokenization/casing; the paper reports true-cased BLEU without punctuation, so its score is not directly comparable without matching that normalization.
- Fairness: evaluate the same held-out audio/reference candidates for both systems. Do not select or tune models on test outputs.

NLLB's model card lists Tunisian Arabic as `aeb_Arab` and English as `eng_Latn`; its documented inference is local Transformers usage and the card marks hosted inference unavailable. Its broad multilingual coverage does not guarantee reliable Tunisian dialect translation. Candidate bilingual alignment must be manually inspected. Whisper-small is Apache-2.0; NLLB is CC-BY-NC-4.0 and is treated as research-only/non-commercial. Model revisions are pinned in the experiment configuration.

## 5. Experimental setup

Random seed: 42. The project adaptation configuration uses five epochs, batch size 1, and gradient accumulation 8; this is a compute-conscious experiment and not a reproduction of the paper's 80-epoch schedule. The smoke-test configuration is one epoch with no more than 16 training and 8 evaluation segments. Defaults are opt-in. Save package versions, model IDs/revisions, device, model size, timing, configuration, predictions, and metrics to `results/`.

The development environment has Python 3.12.2, `torch 2.6.0+cpu` (`torch.cuda.is_available() == False`), an NVIDIA RTX 3050 Laptop GPU with 4 GiB VRAM (driver 511.65; CUDA 11.6 reported by `nvidia-smi`), and a 12th Gen Intel Core i5-12500H. The installed PyTorch build cannot use that GPU. `transformers`, `datasets`, and `sacrebleu` are not installed. All code cells of both notebooks were executed in order with model runs disabled: metadata loading, text EDA, bilingual candidate alignment, and audio inventory reporting completed. No ASR model or translation model was loaded or run.

## 6. Results

### 6.1 Project-run results

| Experiment | Status | Result |
|---|---|---|
| Metadata schema and split inspection | Completed | See `results/metrics/dataset_inspection.json` |
| Notebook metadata/EDA/alignment cells | Completed | Both notebooks executed top-to-bottom with model cells disabled |
| Authorized local WAV audit | Completed, blocked | 0/108 expected WAVs present; see `results/metrics/audio_inventory.json` |
| ASR metadata EDA figure | Completed | `results/figures/asr_metadata_eda.png`; derived from transcript metadata only |
| ASR zero-shot and fine-tuning | Not run | Audio is not hosted in the dataset repository |
| Direct speech translation | Not run | Audio is not hosted in the dataset repository |
| ASR→MT pipeline | Not run | Audio and model dependencies are unavailable |

No project WER, CER, BLEU, chrF, training curve, inference-time, or memory figure is claimed.

### 6.2 Paper-reported comparison (context only)

| Paper system | Test metric |
|---|---:|
| Whisper large-v3 zero-shot ASR | WER 94.00%, CER 67.90% |
| Fine-tuned Whisper large-v3 ASR | WER 25.19%, CER 12.33% |
| Fine-tuned w2v-BERT-2.0T ASR | WER 21.37%, CER 8.34% |
| Whisper large-v3 zero-shot speech translation | BLEU 13.95 |
| Fine-tuned Whisper large-v3 speech translation | BLEU 25.68 |

These are copied from Bougares et al. (2025), not reproduced outcomes.

## 7. Error analysis

No model hypotheses are available to analyze. Upon execution, inspect the largest WER/CER and translation errors, code-switched utterances, names, disfluencies, short segments, and normalization-sensitive cases. Save anonymization-safe examples with references and predictions; do not cherry-pick only successful outputs.

## 8. Limitations

The dataset spans 108 talks but does not represent every Tunisian region or speaker group. Orthographic conventions are not universal. Audio availability and licensing constrain reproducibility: the repository provides URLs, not audio files, and the license is non-commercial/no-derivatives. CSV alignment is not fully exact and includes blanks and a version mismatch relative to the paper. Speaker-level split isolation cannot be verified from the exposed CSV schema. The local GPU is memory-limited, and no experimental scores are currently available.

## 9. Conclusion and next steps

The project has an auditable metadata inspection path, conservative bilingual candidate pairing, shared scoring helpers, and two experiment notebooks. The next steps are to obtain permitted audio, manually verify bilingual candidates, install a compatible PyTorch/Hugging Face environment, run the smoke test, and then run and record the full train/dev/test experiments. Do not claim model improvements until held-out results are measured.

## References

Bougares, F., Mdhaffar, S., Elleuch, H., & Estève, Y. (2025). TEDxTN: A Three-way Speech Translation Corpus for Code-Switched Tunisian Arabic - English. *Proceedings of the Third Arabic Natural Language Processing Conference*, 278–287. https://doi.org/10.18653/v1/2025.arabicnlp-main.22

Radford, A. et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. https://arxiv.org/abs/2212.04356

NLLB Team et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation. https://arxiv.org/abs/2207.04672

TEDxTN dataset: https://huggingface.co/datasets/fbougares/TEDxTN

Whisper-small model card: https://huggingface.co/openai/whisper-small

NLLB-200 distilled 600M model card: https://huggingface.co/facebook/nllb-200-distilled-600M
