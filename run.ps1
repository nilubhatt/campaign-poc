<#
  Campaign Intelligence (lean) - Windows launcher.

  Usage:
    .\run.ps1 -Setup                 # one-time: venv + deps + auto-install Ollama + pull model
    .\run.ps1                        # run the HTTP server on demand (foreground)
    .\run.ps1 -Mode service          # register + start as a background service (Task Scheduler, at logon)
    .\run.ps1 -Mode service -Stop    # stop + unregister the background service
    .\run.ps1 -HostAddr 0.0.0.0 -Port 8086   # bind for a tunnel (Claude Web); default is local-only

  Notes:
    -Setup downloads the official Ollama Windows installer if Ollama is missing (needs internet once).
    Default bind is 127.0.0.1 (no firewall prompt). Use -HostAddr 0.0.0.0 only when exposing via a tunnel.
#>
[CmdletBinding()]
param(
  [switch]$Setup,
  [ValidateSet('ondemand','service')] [string]$Mode = 'ondemand',
  [switch]$Stop,
  [string]$HostAddr = '127.0.0.1',
  [int]$Port = 8086,
  [ValidateSet('ollama','hash','voyage')] [string]$EmbedProvider = 'ollama'
)

$ErrorActionPreference = 'Stop'
$Root      = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv      = Join-Path $Root '.venv'
$VenvPy    = Join-Path $Venv 'Scripts\python.exe'
$TaskName  = 'CampaignIntelligenceServer'

function Find-Python {
  foreach ($c in @('py -3','python','python3')) {
    try { & $c.Split(' ')[0] $c.Split(' ')[1..9] --version *> $null; if ($LASTEXITCODE -eq 0) { return $c } } catch {}
  }
  throw "Python 3 not found. Install it from https://www.python.org/downloads/ (check 'Add to PATH')."
}

function Ensure-Ollama {
  if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "Ollama already installed." -ForegroundColor Green
  } else {
    Write-Host "Ollama not found - downloading the official Windows installer..." -ForegroundColor Yellow
    $inst = Join-Path $env:TEMP 'OllamaSetup.exe'
    Invoke-WebRequest -Uri 'https://ollama.com/download/OllamaSetup.exe' -OutFile $inst
    Write-Host "Installing Ollama silently..."
    Start-Process -FilePath $inst -ArgumentList '/VERYSILENT','/NORESTART' -Wait
    $env:Path += ';' + (Join-Path $env:LOCALAPPDATA 'Programs\Ollama')
  }
  # start the service and pull the embedding model
  Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 3
  Write-Host "Pulling embedding model nomic-embed-text..." -ForegroundColor Cyan
  & ollama pull nomic-embed-text
}

function Do-Setup {
  $py = Find-Python
  if (-not (Test-Path $VenvPy)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    & $py.Split(' ')[0] $py.Split(' ')[1..9] -m venv $Venv
  }
  Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
  & $VenvPy -m pip install --upgrade pip
  & $VenvPy -m pip install -r (Join-Path $Root 'requirements.txt')
  Ensure-Ollama
  Write-Host "`nSetup complete. Start the server with:  .\run.ps1" -ForegroundColor Green
}

function Start-Server {
  if (-not (Test-Path $VenvPy)) { throw "Not set up yet. Run:  .\run.ps1 -Setup" }
  $env:CAMPAIGN_POC_EMBED_PROVIDER = $EmbedProvider
  $env:CAMPAIGN_POC_HOST = $HostAddr
  $env:CAMPAIGN_POC_PORT = "$Port"
  Write-Host "Serving http://${HostAddr}:${Port}  (/mcp, /upload, /healthz)" -ForegroundColor Green
  & $VenvPy -m http_app
}

function Register-Service {
  # Background "service" via Task Scheduler: runs hidden at logon and starts now.
  $action  = New-ScheduledTaskAction -Execute $VenvPy -Argument '-m http_app' -WorkingDirectory $Root
  $trigger = New-ScheduledTaskTrigger -AtLogOn
  $set     = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
  $envWrap = "cmd /c set CAMPAIGN_POC_EMBED_PROVIDER=$EmbedProvider&& set CAMPAIGN_POC_HOST=$HostAddr&& set CAMPAIGN_POC_PORT=$Port&& `"$VenvPy`" -m http_app"
  $action  = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c `"$envWrap`"" -WorkingDirectory $Root
  Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $set -Force | Out-Null
  Start-ScheduledTask -TaskName $TaskName
  Write-Host "Background service '$TaskName' registered and started (runs at logon)." -ForegroundColor Green
  Write-Host "For a hardened Windows Service instead, wrap http_app with NSSM or WinSW (see WINDOWS.md)."
}

function Unregister-Service {
  Stop-ScheduledTask  -TaskName $TaskName -ErrorAction SilentlyContinue
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
  Write-Host "Background service '$TaskName' stopped and removed." -ForegroundColor Green
}

if ($Setup)                     { Do-Setup }
elseif ($Mode -eq 'service' -and $Stop) { Unregister-Service }
elseif ($Mode -eq 'service')    { Register-Service }
else                            { Start-Server }
