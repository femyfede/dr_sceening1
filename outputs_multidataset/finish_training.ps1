# Finish the DR pipeline: merge worker features, split train/test, run training.
# Usage: powershell -ExecutionPolicy Bypass -File outputs_multidataset\finish_training.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root "venv\Scripts\python.exe"

Write-Host "== merging worker files =="
& $py (Join-Path $PSScriptRoot "run_extract.py") --merge
if ($LASTEXITCODE -ne 0) { throw "merge failed" }

Write-Host "`n== training =="
& $py (Join-Path $PSScriptRoot "multidataset\train.py") --config (Join-Path $PSScriptRoot "multidataset\config.yaml")
if ($LASTEXITCODE -ne 0) { throw "training failed" }

Write-Host "`nDone. Artifacts in: $PSScriptRoot (model_bundle.joblib, plots\, metrics_summary.csv)"
