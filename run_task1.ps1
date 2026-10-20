# Task 1 (DPO) overnight runner.  From the repo root:
#   powershell -ExecutionPolicy Bypass -File .\run_task1.ps1
# Each step is skipped if its output already exists (the python scripts check), so re-running after a
# crash resumes where it stopped. Full console output goes to results\task1_dpo\run_task1_log.txt.

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$Py  = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$Log = Join-Path $PSScriptRoot "results\task1_dpo\run_task1_log.txt"
New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

function Step([string]$title, [string[]]$argv, [string]$skipIf = "") {
    if ($skipIf -and (Test-Path $skipIf)) { "`n=== SKIP (done): $title" | Tee-Object -FilePath $Log -Append | Out-Host; return $true }
    "`n=== $title  [$(Get-Date -Format 'HH:mm:ss')]" | Tee-Object -FilePath $Log -Append | Out-Host
    & $Py @argv 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $Log -Append | Out-Host
    $ok = ($LASTEXITCODE -eq 0)
    "=== $(if ($ok) {'DONE'} else {"FAILED (exit $LASTEXITCODE)"}): $title  [$(Get-Date -Format 'HH:mm:ss')]" | Tee-Object -FilePath $Log -Append | Out-Host
    return $ok
}

"##### Task 1 run started $(Get-Date)" | Tee-Object -FilePath $Log -Append
$null = Step "Objective unit tests" @("-m", "tests.test_objectives")
$std = Step "Train standard DPO (1 epoch)" @("-m", "task1_dpo.train", "--config", "configs/dpo.yaml", "--run-name", "standard") "outputs\task1_dpo\standard\adapter_model.safetensors"
$null = Step "Evaluate SFT baseline" @("-m", "task1_dpo.evaluate", "--config", "configs/dpo.yaml", "--adapter", "none", "--name", "sft") "results\task1_dpo\sft\eval_dpo_standard_eval.json"
if ($std) {
    $null = Step "Evaluate standard DPO" @("-m", "task1_dpo.evaluate", "--config", "configs/dpo.yaml", "--adapter", "outputs/task1_dpo/standard", "--name", "standard") "results\task1_dpo\standard\eval_dpo_standard_eval.json"
}
$null = Step "Beta study (train + evaluate 3 short forks)" @("-m", "task1_dpo.ablate_beta", "--config", "configs/dpo.yaml")
if ($std) {
    $null = Step "Length study (train length-balanced + stratified evals)" @("-m", "task1_dpo.analyze_length", "--config", "configs/dpo.yaml")
}
"##### Task 1 run finished $(Get-Date)" | Tee-Object -FilePath $Log -Append
