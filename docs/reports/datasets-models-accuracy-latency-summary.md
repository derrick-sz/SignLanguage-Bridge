# Sign Language Bridge: Datasets, Models, Accuracy, and Latency

**Updated:** October 23, 2026

## Executive summary

The project currently has two main recognition pipelines:

- **Chinese Sign Language (CE-CSL):** a video-to-gloss sequence model evaluated on 500 official test clips. Its corpus-level word error rate (WER) is **32.80%**.
- **Brazilian Sign Language (Libras):** a Random Forest classifier for a 12-class pilot. It reached **63.89% accuracy** and a **0.6034 macro-F1** on 72 test clips.

In recent video-file replay, recognition results returned **0.27–0.40 seconds** after a CSL window ended and **1.26 seconds** after a Libras window ended. These figures exclude the time needed to collect each window. They are not end-to-end camera latency.

There is no measured camera-based recognition accuracy or end-to-end P95 latency to report. The camera was tested for capture and preview, but that run did not produce a valid recognition result.

## Datasets and models

| Language and dataset | Dataset size and split | Model and input | Current scope |
|---|---|---|---|
| **Chinese Sign Language — CE-CSL** | 5,988 clips: 4,973 training, 515 development, and 500 test clips. The vocabulary contains 3,515 glosses plus the CTC blank token. Signers A–L appear across splits, so performance on unseen signers has not been established. | TFNet processes RGB video and produces a gloss sequence using CTC decoding. The ONNX model accepts complete clips, up to 600 frames or 15 seconds. | Complete-clip sequence recognition. The model is bidirectional; it does not provide native causal, frame-by-frame decoding. Its output is a gloss sequence, not a natural-language translation. |
| **Brazilian Sign Language — Libras** | 5,347 MP4 files across 178 source items. The current pilot uses 360 clips from 12 classes, split into 216 training, 72 validation, and 72 test clips. | MediaPipe Holistic extracts 225-dimensional body and hand landmarks at 30 frames per clip. A 300-tree Random Forest classifies the features and currently runs on CPU. | Finite-clip classification for 12 classes. This is not a result for all 178 items or for sentence translation. |

**Terminology:** TFNet is the recognition network. ONNX is its exported model format and runtime graph. CTC describes how frame-level token scores are turned into a sequence; this pipeline uses prefix-beam decoding. “TFNet (ONNX deployment, CTC decoding)” therefore names one recognition pipeline, not three separate models.

| Model | Evaluation set | Recorded result | How to interpret it |
|---|---|---|---|
| **CE-CSL TFNet (ONNX deployment, CTC decoding)** | 500 official test clips; 3,040 reference words | Corpus WER: **32.7961%**. Mean utterance WER: **32.4394%**. Edit distance: 997. | Report WER as an error rate. It is not classification accuracy, and 1 − WER should not be presented as accuracy. |
| **Libras MediaPipe225 + Random Forest** | 72 test clips across 12 classes | **46/72 correct**; accuracy: **63.89%**; macro-F1: **0.6034**; coverage: **100%**. | This is a small, finite-clip pilot. The split groups examples by local informant within each item; generalization to globally unseen signers and rejection of unknown signs have not been validated. |

These results come from different tasks and metrics, so their percentages should not be compared directly.

### Experimental results — not current product metrics

- A frozen CE-CSL visual representation with a Ridge classification head scored **72/72 (100%)** on the same 12-class Libras test subset. A random-initialized visual representation with the same type of head scored **17/72 (23.61%)**. This is a limited transfer experiment, not the deployed Libras model; it does not establish performance across all 178 items or on unseen signers.
- A Chinese T1-feature + TCN + CTC experiment recorded **77.24% test WER**, with substantial omissions. It did not provide a quality basis for replacing TFNet.

## Recognition latency

### Complete clips and rolling windows

| Pipeline and measurement | Recorded latency | Measurement boundary |
|---|---:|---|
| **CE-CSL TFNet (ONNX Runtime on CUDA) — offline 500-clip evaluation** | Median model inference: **0.395 s**. Median total per clip, including preprocessing and decoding: **1.328 s**. | Processing of an existing complete video file. Excludes camera capture and the time spent recording the clip. These are local measurements, not a cross-device service guarantee. |
| **CE-CSL TFNet on CPU — one 8.2-second, 247-frame clip** | **6.31–6.62 s** over three runs using four threads. | A single-clip CPU comparison, not a test-set latency statistic. |
| **CSL rolling-window video replay** | **0.265–0.395 s**; five results were displayed. | Time from the window's final frame entering the process to the inference result. Includes preprocessing, but excludes the time needed to accumulate the window. Some ONNX operations ran on CPU. |
| **Libras rolling-window video replay** | **1.259 s**; one result was displayed. | Time from the window's final frame entering the process to the result, including landmark processing. A second call was cancelled at end-of-file. MediaPipe feature extraction was the main cost. |

### Camera and continuous-recognition limits

A physical-camera recovery check completed two rounds of capture and preview, observing 211 frames in total. It did not produce a valid action candidate or recognition result. Therefore, there is no measured camera-recognition latency, camera-recognition accuracy, or end-to-end P95 to report. The rolling-window figures above came from replaying real video files; they are not camera-to-result timings.

The continuous mode repeatedly applies existing clip models to short video windows and shows editable candidates. This demonstrates the window-scheduling and result-return path. It does not establish native streaming recognition, continuous sentence translation, or formal subtitle accuracy.

## Presentation-ready summary

> The project has two main recognition pipelines: Chinese Sign Language sequence recognition and a 12-class Libras clip classifier. TFNet achieved a 32.80% corpus WER on the 500-clip CE-CSL test set. The Libras Random Forest achieved 63.89% accuracy on 72 test clips. In real video-file replay, CSL window results returned about 0.27–0.40 seconds after a window ended, while Libras results took about 1.26 seconds. These figures exclude window-collection time and are not end-to-end camera measurements. The Libras score does not cover all 178 source items, and the CSL WER and Libras accuracy are different metrics.



