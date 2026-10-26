# E00 audit helper #2: SHA256 for SLRT TwoStream family weights (F: drive + Paper-algorithm copy).
# Read-only. Output: experiments/p1/e00/_hash_raw_slrt.txt
$ErrorActionPreference = 'Stop'
$root = 'E:\githubproject\SignLanguage-Bridge'
$out = Join-Path $root 'experiments\p1\e00\_hash_raw_slrt.txt'
$files = @(
  'F:\model_weigth\ckpts\TwoStreamSLT\TwoStream\csl-daily_joint\ckpts\best.ckpt',
  'F:\model_weigth\ckpts\TwoStreamSLT\TwoStream\csl-daily_keypoint\ckpts\best.ckpt',
  'F:\model_weigth\ckpts\TwoStreamSLT\TwoStream\csl-daily_video\ckpts\best.ckpt',
  'F:\model_weigth\ckpts\TwoStreamSLT\TwoStream\csl-daily_s2g\ckpts\best.ckpt',
  'F:\model_weigth\ckpts\TwoStreamSLT\SingleStream\csl-daily_s2t\ckpts\best.ckpt',
  'F:\model_weigth\ckpts\TwoStreamSLT\SingleStream\csl-daily_s2g\ckpts\best.ckpt',
  'E:\githubproject\SignLanguage-Bridge\Paper-algorithm\SLRT-main\Online\CTC_fusion\results\csl-daily_s2g\ckpts\best.ckpt'
)
$lines = New-Object System.Collections.Generic.List[string]
foreach ($p in $files) {
  if (-not (Test-Path -LiteralPath $p)) { $lines.Add("MISSING|0|$p"); continue }
  $fi = Get-Item -LiteralPath $p
  $h = (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()
  $lines.Add("$h|$($fi.Length)|$p")
}
Set-Content -LiteralPath $out -Value $lines -Encoding UTF8
Write-Output "DONE $($lines.Count) lines -> $out"
