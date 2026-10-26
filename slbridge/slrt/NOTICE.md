# SLRT source attribution

The S3D primitives in `_s3d_base.py` are extracted (AST formatting only) from
FangyunWei/SLRT, `Online/CSLR/modelling/S3D_base.py`:
https://github.com/FangyunWei/SLRT/blob/main/Online/CSLR/modelling/S3D_base.py

`network.py` implements the configured inference path from `S3D.py`,
`two_stream.py`, `fusion.py`, `Visualhead.py`, and `recognition.py` in that same
directory. Unused training, pyramid, attention, loss and K400 initialization
paths are omitted. Parameter paths are preserved for strict checkpoint loading.

Source authors: Ronglai Zuo, Fangyun Wei, Brian Mak. Paper: *Towards Online
Continuous Sign Language Recognition and Translation*, EMNLP 2024,
https://aclanthology.org/2024.emnlp-main.619/ .

The local upstream snapshot does not include a root LICENSE file. This notice
records attribution and does not grant a new license for upstream code,
checkpoints or datasets. Source hashes and numerical equivalence verification
are recorded in the dated integration report/artifacts.
