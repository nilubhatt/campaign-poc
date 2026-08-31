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

$zip = "campaign-intelligence-windows-amd64.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path dist\campaign-intelligence\* -DestinationPath $zip
Write-Host "Bundle: $zip"
Write-Host "Run:    dist\campaign-intelligence\campaign-intelligence.exe serve   (or 'stdio')"
