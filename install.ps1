$ErrorActionPreference = "Stop"
# GelTilt one-click installer: venv -> runtime/test deps -> editable install ->
# import self-check -> full pytest. Any failed command aborts the script.
Set-Location -Path $PSScriptRoot

function Assert-ExitCode {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) { throw "$Step failed with exit code $LASTEXITCODE" }
}

$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }

Write-Host "[1/6] Check Python (>= 3.10)" -ForegroundColor Cyan
& $py -c "import sys; assert sys.version_info >= (3,10), sys.version; print(sys.version)"
Assert-ExitCode "[1/6] interpreter check"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[2/6] Create virtual environment .venv" -ForegroundColor Cyan
    & $py -m venv .venv
    Assert-ExitCode "[2/6] create venv"
} else {
    Write-Host "[2/6] Reuse existing virtual environment .venv" -ForegroundColor Cyan
}
$venvPy = Join-Path $PWD ".venv\Scripts\python.exe"

Write-Host "[3/6] Install runtime and test dependencies" -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip
Assert-ExitCode "[3/6] upgrade pip"
& $venvPy -m pip install -r requirements.txt
Assert-ExitCode "[3/6] install requirements"

Write-Host "[4/6] Install this project in editable mode" -ForegroundColor Cyan
& $venvPy -m pip install -e .
Assert-ExitCode "[4/6] editable install"

Write-Host "[5/6] Import self-check" -ForegroundColor Cyan
& $venvPy -c "import numpy, PIL, cv2; print(numpy.__version__, PIL.__version__, cv2.__version__)"
Assert-ExitCode "[5/6] import check"
& $venvPy -m geltilt --help
Assert-ExitCode "[5/6] geltilt --help"

Write-Host "[6/6] Run unit tests" -ForegroundColor Cyan
& $venvPy -m pytest -q
Assert-ExitCode "[6/6] pytest"

Write-Host "Done. Start the app with: .venv\Scripts\python.exe run_app.py" -ForegroundColor Green
