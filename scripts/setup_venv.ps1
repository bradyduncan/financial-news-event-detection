$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $projectRoot ".venv"
$requirementsPath = Join-Path $projectRoot "requirements.txt"

if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
}

$activate = Join-Path $venvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    throw "Virtual environment activation script not found: $activate"
}

& $activate
python -m pip install --upgrade pip
python -m pip install -r $requirementsPath

Write-Host "Venv ready: $venvPath"
Write-Host "Activate with: $activate"
