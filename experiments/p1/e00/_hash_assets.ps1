# E00 audit helper: compute SHA256 + bytes for model assets and data references.
# Read-only. Output: experiments/p1/e00/_hash_raw.txt, one line per file: sha256|bytes|path
$ErrorActionPreference = 'Stop'
$root = 'E:\githubproject\SignLanguage-Bridge'
$out = Join-Path $root 'experiments\p1\e00\_hash_raw.txt'
$files = @(
  'models\external\uni_sign\csl_daily_rgb_pose_slt.pth',
  'models\external\uni_sign\csl_daily_pose_only_slt.pth',
  'models\external\uni_sign\mt5-base\pytorch_model.bin',
  'models\external\uni_sign\faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth',
  'models\external\uni_sign\hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth',
  'models\external\uni_sign\rtmlib-cache\hub\checkpoints\rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx',
  'models\external\uni_sign\rtmlib-cache\hub\checkpoints\yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx',
  'models\external\.downloads\Uni-Sign-eed438bcb49e30405cd6ccdfcccca330c134e830.zip',
  'models\external\EMNLP24_Online\CSL-Daily\ckpts\cslr_best.ckpt',
  'models\external\SLRT_Online\Online\CSLR\results\csl-daily_ISLR\ckpts\best.ckpt',
  'models\external\SLRT_Online\pretrained_models\s3ds_actioncls_ckpt',
  'models\external\SLRT_Online\pretrained_models\mBart\pytorch_model.bin',
  'models\external\SLRT_Online\pretrained_models\mBART_zh_fullvocab\pytorch_model.bin',
  'models\external\SLRT_Online\pretrained_models\mBART_zh_fullvocab\gloss2ids.pkl',
  'models\external\SLRT_Online\pretrained_models\mBART_zh_fullvocab\gloss_embeddings.bin',
  'models\external\SLRT_Online\pretrained_models\mBART_zh_gloss_assets\gloss2ids.pkl',
  'experiments\p1\datasets\manifest_1086.csv',
  'experiments\p1\datasets\pilot_50.csv',
  'experiments\p1\references\gt_1086.csv',
  'experiments\p1\references\pilot_50_diagnostic_EVAL_ONLY.csv',
  'F:\model_weigth\data\csl-daily\csl-daily-keypoints.pkl',
  'F:\model_weigth\data\csl-daily\csl-daily.test',
  'F:\model_weigth\data\csl-daily\gloss2ids.pkl',
  'models\external\uni_sign_source\out\eval_csl_daily_pose_only\test_tmp_refs.txt'
)
$lines = New-Object System.Collections.Generic.List[string]
foreach ($rel in $files) {
  $p = if ([System.IO.Path]::IsPathRooted($rel)) { $rel } else { Join-Path $root $rel }
  if (-not (Test-Path -LiteralPath $p)) { $lines.Add("MISSING|0|$rel"); continue }
  $fi = Get-Item -LiteralPath $p
  $h = (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()
  $lines.Add("$h|$($fi.Length)|$rel")
}
Set-Content -LiteralPath $out -Value $lines -Encoding UTF8
Write-Output "DONE $($lines.Count) lines -> $out"
