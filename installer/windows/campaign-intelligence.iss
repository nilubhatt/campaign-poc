; Inno Setup script - end-user Windows installer for Campaign Intelligence (lean).
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
AppVersion=0.2.7
DefaultDirName={autopf}\CampaignIntelligence
DefaultGroupName={#AppName}
OutputBaseFilename=CampaignIntelligence-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
UninstallDisplayName={#AppName}

[Files]
Source: "..\..\dist\campaign-intelligence\*"; DestDir: "{app}"; Excludes: "models\*"; Flags: recursesubdirs createallsubdirs
; The CLIP checkpoint is ~303MB of near-random tensor data: LZMA2/max spends minutes on it
; for ~0% gain, and in a solid stream every install pays to decompress it in order. Kept
; out of the solid block and stored uncompressed.
Source: "..\..\dist\campaign-intelligence\models\*"; DestDir: "{app}\models"; Flags: recursesubdirs createallsubdirs nocompression solidbreak

[Icons]
Name: "{group}\{#AppName} (server)"; Filename: "{app}\{#AppExe}"; Parameters: "serve"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"

[Tasks]
Name: "ollama";  Description: "Install Ollama and download the embedding model (needs internet)"; GroupDescription: "Dependencies:"
Name: "desktop"; Description: "Connect Claude Desktop automatically (stdio)"; GroupDescription: "Integration:"
Name: "service"; Description: "Run the server in the background at logon"; GroupDescription: "Startup:"; Flags: unchecked

[Run]
; Only the step whose failure genuinely does not matter stays here. Everything the product
; needs in order to WORK moved into [Code] below, because an entry in this section cannot
; stop the install: Inno ignores a [Run] entry's exit code, and these execute after
; ssPostInstall and immediately before the success page. That ordering is also why the
; self-test cannot simply live in ssPostInstall - it would check a machine before Ollama
; had been installed and report a failure the installer had not finished causing.
;
; Background service via Task Scheduler (runs at logon). Genuinely optional.
Filename: "schtasks.exe"; \
  Parameters: "/Create /F /SC ONLOGON /TN CampaignIntelligenceServer /TR ""'{app}\{#AppExe}' serve"""; \
  StatusMsg: "Registering background service..."; Tasks: service; Flags: runhidden

[UninstallRun]
Filename: "schtasks.exe"; Parameters: "/Delete /F /TN CampaignIntelligenceServer"; Flags: runhidden; RunOnceId: "DelTask"

[Code]
// Item 4.1: a post-install self-test that BLOCKS the success screen, naming the component
// that is not working.
//
// The product review's machine had this product installed with visual search silently dead,
// and the only way to discover it was a 60-second timeout inside a tool call followed by a
// read of the server's source. An installer that reports success over that state is the
// defect - so this one refuses to.

function RunAndCapture(Cmd, Params: String; var Output: String): Integer;
var
  LogPath: String;
  Code: Integer;
  Lines: TArrayOfString;
  I: Integer;
begin
  LogPath := ExpandConstant('{tmp}\selftest.txt');
  // cmd.exe rather than Exec on the binary directly, because the output has to be captured:
  // the criterion is naming the failing component, and a bare exit code names nothing.
  if not Exec(ExpandConstant('{cmd}'), '/C ""' + Cmd + '" ' + Params + '" > "' + LogPath + '" 2>&1',
              '', SW_HIDE, ewWaitUntilTerminated, Code) then
  begin
    Output := 'the self-test could not be started';
    Result := -1;
    exit;
  end;
  Output := '';
  if LoadStringsFromFile(LogPath, Lines) then
    for I := 0 to GetArrayLength(Lines) - 1 do
      Output := Output + Lines[I] + #13#10;
  Result := Code;
end;

procedure InstallOllama();
var
  Code: Integer;
  Script: String;
begin
  if not WizardIsTaskSelected('ollama') then exit;
  Script := '-NoProfile -ExecutionPolicy Bypass -Command "' +
            'if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { ' +
            '$i = \"$env:TEMP\OllamaSetup.exe\"; ' +
            'Invoke-WebRequest https://ollama.com/download/OllamaSetup.exe -OutFile $i; ' +
            'Start-Process $i -ArgumentList ''/VERYSILENT'',''/NORESTART'' -Wait }; ' +
            'Start-Process ollama -ArgumentList ''serve'' -WindowStyle Hidden; ' +
            'Start-Sleep 3; ollama pull nomic-embed-text"';
  // A failure here is deliberately not fatal on its own. Whether it actually broke anything
  // is the self-test's question to answer, and it answers it about the machine rather than
  // about whether one command returned zero (item 4.3).
  Exec('powershell.exe', Script, '', SW_HIDE, ewWaitUntilTerminated, Code);
end;

procedure ConfigureDesktop();
var
  Code: Integer;
begin
  if not WizardIsTaskSelected('desktop') then exit;
  Exec(ExpandConstant('{app}\{#AppExe}'), 'configure-desktop', '', SW_HIDE,
       ewWaitUntilTerminated, Code);
end;

procedure SelfTest();
var
  Output: String;
  Code: Integer;
begin
  Code := RunAndCapture(ExpandConstant('{app}\{#AppExe}'), 'health-check', Output);
  if Code = 0 then exit;
  // RaiseException is what actually blocks the success page: Inno reports the install as
  // not completed and shows this text. A MsgBox here would be dismissed and the success
  // screen shown anyway, which is the behaviour being fixed.
  RaiseException(
    'Campaign Intelligence was installed but is not working, so setup has not completed.' +
    #13#10#13#10 + Output + #13#10 +
    'The files are in ' + ExpandConstant('{app}') + '. Fix what is named above and run:' +
    #13#10 + '  "' + ExpandConstant('{app}\{#AppExe}') + '" health-check');
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // Order matters: the self-test has to run against the finished machine, which is why
    // these three moved out of [Run] (see the comment there).
    InstallOllama();
    ConfigureDesktop();
    SelfTest();
  end;
end;
