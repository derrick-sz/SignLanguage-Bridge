---
task_id: d
role: Lead technical and evidence reviewer
objective: Check learning/annotation opportunities and limits of offline recognition, and spot-check decision-driving originals
status: complete
confidence: medium
sources_found: 4
acceptance_met: yes
---

## Sources

- [D1] Battisti & Ebling, Automatic Annotation Elaboration as Feedback to Sign Language Learners | https://aclanthology.org/2024.law-1.5.pdf | PRIMARY EXPERIMENT | 2024-03 | Aut:9 Rec:7 Rel:8 Dep:8 = 8.0; expert annotation-to-text study, not end-to-end camera error detection or learning-gain trial.
- [D2] ELAN, The Language Archive / Max Planck Institute | https://archive.mpi.nl/tla/elan | OFFICIAL SOFTWARE DESCRIPTION | undated current page checked 2026-10-11 | Aut:9 Rec:8 Rel:9 Dep:7 = 8.25; proves existing annotation features, not demand for our tool.
- [D3] Mukushev et al., FluentSigners-50 | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0273649 | ORIGINAL DATASET STUDY | 2022-09 | Aut:9 Rec:6 Rel:8 Dep:9 = 8.0; KRSL dataset, methodological relevance, not transferable model accuracy.
- [D4] Zuo, Wei & Mak, Towards Online Continuous Sign Language Recognition and Translation | https://aclanthology.org/2024.emnlp-main.619/ | ORIGINAL MODEL STUDY | 2024-11 | Aut:9 Rec:7 Rel:9 Dep:8 = 8.25; model design and benchmark proof, not validation on this project's camera inputs.

Scores are screening judgments. Root also re-opened A3 Chinese creator methods/design/limitations, A1 author PDF, A2 WFD report, B3 current terms, B1 app description, B7/B8 product descriptions and B9 deployment PDF. Significant claims are checked against originals, not solely summarized notes.

## Findings (facts only)

- D1 uses 100 expert-annotated DSGS learner sentences and five beginner/intermediate volunteer evaluators. GPT-4 elaborates existing error annotations into readable German feedback; analysis found omitted/misrepresented non-manual errors, invented extra information and references to nonexistent example videos. It does not demonstrate automatic error detection directly from webcam input or longitudinal learning gain. [D1, sections 3–5]
- ELAN supports time-aligned multilayer annotations including gloss, translation and comments; documents use EAF XML and the software source is GPL3. [D2]
- FluentSigners-50 deliberately defines signer-independent, age-independent and unseen-sentence evaluation splits and collects variation in signing, devices and backgrounds. It is KRSL and not a substitute for local CSL/Libras evaluation. [D3]
- EMNLP2024's online method develops a sign dictionary, trains an isolated recognition model and uses sliding windows; its abstract separates offline whole-video CTC from this specifically trained online approach. [D4]
- Current local CTC manifest `artifacts/models/cecsl-tfnet-onnx-20261009/manifest.json:30-32` limits complete input to max_frames=600 and max_seconds=15.0. Root and independent reviewer inspected the actual file. A long-video review pilot needs manually bounded model requests plus source-time preservation; it cannot rely on unaccepted automatic endpoint quality.

## Analysis (our synthesis)

- Learning is a credible secondary direction, but recognition confidence or DTW closeness is not a linguistic correctness score. Start with recordings/reference replay and teacher annotations; a bounded LLM could explain verified annotations. [D1]
- A new annotation workbench should support EAF exchange or a narrow review layer alongside ELAN rather than recreate every generic editing feature. Useful differentiation to test: model suggestions, source evidence, dedup/quality flags and bilingual review/versioning. [D2 + local C]
- The product test must independently measure new signers, unseen/background movements, repair cost and other-party comprehension. A fast known-video benchmark supports compute readiness only. [D3 + local C9/C10]
- Incremental inference can be investigated in parallel; slicing the current offline CTC input does not establish the trained online model or its quality. [D4]

## Support snippets / paraphrases for top claims

- Learning paper PDF pp. 49–53: model receives human error annotations, evaluators rate output explanations, authors record missing/incorrect non-manual error information and invented video references. Supports bounded annotation explanation, not automatic teacher replacement. [D1]
- ELAN official description: sentence/word/gloss/comment/translation annotations are organized in related tiers and synchronized with source media. Supports standard-compatible exchange as an opportunity. [D2]
- FluentSigners abstract/methods: diverse contributors and recording settings, explicitly separated evaluation protocols. Supports local signer/generalization test design. [D3]
- Online CSLR abstract: offline CTC whole input versus dictionary-trained recognition and sliding windows. Supports keeping inference modes separate. [D4]

## Conflicts / unresolved issues

- SignStudio page contradicts itself on current AI ASL availability; report only adopts the publicly described content workflow, not precise current package coverage.
- Corpus Libras landing page fetched an internal error; Signbank's English page returned navigation only. These are discovered leads, not accepted support for detailed access/licensing claims in final report.

## Leads discovered

- EAF import/export for expert review; teacher-approved error explanations; frozen split by actual participant identity.

## Gaps

- No direct local learning-gain, annotation time-saving or competition usability trial was run. No competitor APIs installed or called.
- Root made 3 initial search query strings and subsequent opens/finds; fetch metrics were not aggregated across agents, so run summary leaves fetch_count unspecified.
