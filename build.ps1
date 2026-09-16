<#
  Build a self-contained Windows bundle (one-folder .exe). Run ON Windows.
  Output: dist\campaign-intelligence\campaign-intelligence.exe (+ _internal\ with the
  bundled sqlite-vec DLL). Produces a .zip you can distribute.
#>
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

py -3 -m venv .venv-build
& .\.venv-build\Scripts\python.exe -m pip install --upgrade pip
# Stamp the build so two local builds are not indistinguishable (section 3.2, defect 10). CI
# writes the same file; without it every locally built binary reports itself identically, and
# calls itself a "source checkout" while being a frozen app.
$sha = (git rev-parse --short HEAD 2>$null)
if (-not $sha) { $sha = "local" }
"$sha $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))" | Set-Content -Encoding ascii build_info.txt

& .\.venv-build\Scripts\python.exe -m pip install -r requirements.txt pyinstaller typer
& .\.venv-build\Scripts\pyinstaller.exe --clean --noconfirm campaign-poc.spec

# The CLIP weights ship inside the bundle so an air-gapped install works untouched. ~303MB (fp16),
# verified by SHA-256. Fetched into .weights-cache first because PyInstaller wipes dist\ on
# every run - fetching straight into it would re-download on every local build.
& .\.venv-build\Scripts\python.exe scripts\fetch_weights.py .weights-cache
if ($LASTEXITCODE -ne 0) { throw "CLIP weights missing or failed verification - refusing to package" }
New-Item -ItemType Directory -Force -Path dist\campaign-intelligence\models | Out-Null
Copy-Item .weights-cache\open_clip_model.safetensors* dist\campaign-intelligence\models\

# Section 12.1: the rulebook, BESIDE the executable rather than only inside _internal\.
# PyInstaller 6 puts every `datas` entry under the contents directory whatever relative
# destination the spec names, while `config.app_dir()` deliberately resolves to the install
# directory and not `sys._MEIPASS` - an administrator has to be able to SEE and EDIT this
# file, which is most of what the item is about. Without this copy every frozen install reads
# the buried one, and edits to the visible file would do nothing, silently.
Copy-Item rulebook.yaml dist\campaign-intelligence\rulebook.yaml
# Section 12.3: the worked example, where an administrator can read it before writing theirs.
Copy-Item docs\example-rulebook.yaml dist\campaign-intelligence\example-rulebook.yaml

$zip = "campaign-intelligence-windows-amd64.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path dist\campaign-intelligence\* -DestinationPath $zip
Write-Host "Bundle: $zip"
Write-Host "Run:    dist\campaign-intelligence\campaign-intelligence.exe serve   (or 'stdio')"
