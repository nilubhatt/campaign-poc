; Inno Setup script — end-user Windows installer for Campaign Intelligence (lean).
; Compile ON Windows: iscc campaign-intelligence.iss  (Inno Setup 6+).
; Expects the PyInstaller bundle already built at ..\..\dist\campaign-intelligence\ (build.ps1).
;
; The installer: copies the bundle to Program Files, optionally installs Ollama + pulls the
; model, wires Claude Desktop automatically (configure-desktop), and can register a background
; service (Task Scheduler). Uninstall removes all of it.

#define AppName "Campaign Intelligence"
#define AppExe  "campaign-intelligence.exe"

[Setup]
AppName={#AppName}
AppVersion=0.2.6
DefaultDirName={autopf}\CampaignIntelligence
DefaultGroupName={#AppName}
OutputBaseFilename=CampaignIntelligence-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
UninstallDisplayName={#AppName}

[Files]
Source: "..\..\dist\campaign-intelligence\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName} (server)"; Filename: "{app}\{#AppExe}"; Parameters: "serve"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"

[Tasks]
Name: "ollama";  Description: "Install Ollama and download the embedding model (needs internet)"; GroupDescription: "Dependencies:"
Name: "desktop"; Description: "Connect Claude Desktop automatically (stdio)"; GroupDescription: "Integration:"
Name: "service"; Description: "Run the server in the background at logon"; GroupDescription: "Startup:"; Flags: unchecked

[Run]
; 1) Ollama (official silent installer) + pull the embedding model.
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -Command ""if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {{ $i=""$env:TEMP\OllamaSetup.exe""; Invoke-WebRequest https://ollama.com/download/OllamaSetup.exe -OutFile $i; Start-Process $i -ArgumentList '/VERYSILENT','/NORESTART' -Wait }}; Start-Process ollama -ArgumentList 'serve' -WindowStyle Hidden; Start-Sleep 3; ollama pull nomic-embed-text"""; \
  StatusMsg: "Installing Ollama and embedding model..."; Tasks: ollama; Flags: runhidden waituntilterminated

; 2) Wire Claude Desktop (stdio -> the installed binary).
Filename: "{app}\{#AppExe}"; Parameters: "configure-desktop"; StatusMsg: "Configuring Claude Desktop..."; Tasks: desktop; Flags: runhidden

; 3) Background service via Task Scheduler (runs at logon).
Filename: "schtasks.exe"; \
  Parameters: "/Create /F /SC ONLOGON /TN CampaignIntelligenceServer /TR ""'{app}\{#AppExe}' serve"""; \
  StatusMsg: "Registering background service..."; Tasks: service; Flags: runhidden

[UninstallRun]
Filename: "schtasks.exe"; Parameters: "/Delete /F /TN CampaignIntelligenceServer"; Flags: runhidden; RunOnceId: "DelTask"
