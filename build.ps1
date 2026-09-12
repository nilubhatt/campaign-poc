<#
  Build a self-contained Windows bundle (one-folder .exe). Run ON Windows.
  Output: dist\campaign-intelligence\campaign-intelligence.exe (+ _internal\ with the
  bundled sqlite-vec DLL). Produces a .zip you can distribute.
#>
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

py -3 -m venv .venv-build
& .\.venv-build\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-build\Scripts\python.exe -m pip install -r requirements.txt pyinstaller typer
& .\.venv-build\Scripts\pyinstaller.exe --clean --noconfirm campaign-poc.spec

# The CLIP weights ship inside the bundle so an air-gapped install works untouched. ~303MB (fp16),
# verified by SHA-256. Fetched into .weights-cache first because PyInstaller wipes dist\ on
# every run - fetching straight into it would re-download on every local build.
& .\.venv-build\Scripts\python.exe scripts\fetch_weights.py .weights-cache
if ($LASTEXITCODE -ne 0) { throw "CLIP weights missing or failed verification - refusing to package" }
New-Item -ItemType Directory -Force -Path dist\campaign-intelligence\models | Out-Null
Copy-Item .weights-cache\open_clip_model.safetensors* dist\campaign-intelligence\models\

$zip = "campaign-intelligence-windows-amd64.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path dist\campaign-intelligence\* -DestinationPath $zip
Write-Host "Bundle: $zip"
Write-Host "Run:    dist\campaign-intelligence\campaign-intelligence.exe serve   (or 'stdio')"
