; Inno Setup script - end-user Windows installer for Campaign Intelligence (lean).
; Compile ON Windows: iscc campaign-intelligence.iss  (Inno Setup 6+).
; Expects the PyInstaller bundle already built at ..\..\dist\campaign-intelligence\ (build.ps1).
;
; The installer: copies the bundle to Program Files, optionally installs Ollama + pulls the
; model, wires Claude Desktop automatically (configure-desktop), and can register a background
; service (Task Scheduler). Uninstall removes all of it.

; Supplied by the build: ISCC.exe /DAppVersion=... reads it from version.py, which is the
; single source. It was hard-coded here and had already drifted - version.py said 0.3.0
; while this said 0.2.7, so Add/Remove Programs, the uninstall entry and upgrade detection
; would every one of them disagree with the binary they had just installed.
#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif
#define AppName "Campaign Intelligence"
#define AppExe  "campaign-intelligence.exe"

[Setup]
AppName={#AppName}
AppVersion={#AppVersion}
; Per-user, which is what this product already is everywhere else: the data directory is
; %LOCALAPPDATA%\CampaignIntelligence (config.py) and Claude Desktop's config is per-user
; (main.py). Installing elevated into Program Files made every post-install step run as
; whoever answered the UAC prompt, so in the case this product is built for - IT installing
; for a marketer - the embedding model was pulled into the ADMIN's Ollama, the ADMIN's Claude
; Desktop was wired, and the self-test passed against an environment the marketer would never
; see. It also removes the UAC prompt, which was creating the impression of a machine-wide
; install that the rest of the product never delivered.
;
; Installing on someone else's behalf therefore means running this while signed in as them.
; No installer can fix that from the other side of a UAC prompt.
DefaultDirName={localappdata}\Programs\CampaignIntelligence
DefaultGroupName={#AppName}
OutputBaseFilename=CampaignIntelligence-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
; Pinned so an upgrade replaces the previous install rather than sitting beside it - without
; it, the move off Program Files would leave the old copy installed and registered.
AppId={{B2F1B0A4-7C3E-4B1E-9E77-CA9D1E7C55A1}
UsePreviousAppDir=yes
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
; stop the install: Inno ignores a [Run] entry's exit code.
;
; The first version of this comment also claimed [Run] executes AFTER ssPostInstall. It does
; not - Setup.MainForm.pas calls ProcessRunEntries (line 234) before SetStep(ssPostInstall)
; (line 241) - so ordering alone would not have required the move. The move is still right,
; for the two reasons above and below: exit codes are ignored here, and the self-test has to
; run before the Claude Desktop wiring rather than after it.
;
; Background service via Task Scheduler (runs at logon). Genuinely optional.
Filename: "schtasks.exe"; \
  Parameters: "/Create /F /SC ONLOGON /TN CampaignIntelligenceServer /TR ""'{app}\{#AppExe}' serve"""; \
  StatusMsg: "Registering background service..."; Tasks: service; Flags: runhidden

[UninstallRun]
Filename: "schtasks.exe"; Parameters: "/Delete /F /TN CampaignIntelligenceServer"; Flags: runhidden; RunOnceId: "DelTask"
; Both Unix uninstallers remove the Claude Desktop connector entry; this one did not, so a
; Windows uninstall left Claude Desktop pointing at a binary that no longer exists. The
; header's claim that uninstall "removes all of it" was false.
Filename: "{app}\{#AppExe}"; Parameters: "unconfigure-desktop"; Flags: runhidden; RunOnceId: "UnwireDesktop"

[Code]
// Item 4.1: a post-install self-test that BLOCKS the success screen, naming the component
// that is not working.
//
// The product review's machine had this product installed with visual search silently dead,
// and the only way to discover it was a 60-second timeout inside a tool call followed by a
// read of the server's source. An installer that reports success over that state is the
// defect - so this one refuses to.

// What the self-test found, if it failed. Not an exception: RaiseException at
// ssPostInstall is caught by Inno itself and discarded --
//
//   Setup.MainForm.pas:  SetStep(ssPostInstall, True);   { HandleExceptions = True }
//     except
//       if HandleExceptions then begin
//         Log('CurStepChanged raised an exception.');
//         Application.HandleException(Self);             { shows it, swallows it }
//
// -- so setup carries on to the Finished page with exit code 0, which is exactly the
// "message dismissed, success screen shown anyway" behaviour this gate exists to replace.
// Only ssInstall re-raises, and an exception there rolls the files back, contradicting the
// advice to fix the problem and re-run against the files just installed.
//
// What IS achievable, and is what the marketer actually needs: do not connect Claude
// Desktop, and replace the success page with the failure. Both are done below.
var
  GSelfTestFailure: String;

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
  // cmd's documented rule with more than two quotes is to strip the FIRST and the LAST.
  // Closing the outer quote before the redirect put the '>' inside quotes, so it was not a
  // redirect at all: the exe received one argument, "health-check > C:\...\selftest.txt",
  // exited 2, and no log file was written - after which LoadStringsFromFile returned False
  // without raising and the dialog named nothing.
  if not Exec(ExpandConstant('{cmd}'), '/C ""' + Cmd + '" ' + Params + ' > "' + LogPath + '" 2>&1"',
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

procedure Say(Message: String);
begin
  // Moving these steps out of [Run] lost their StatusMsg, so the wizard sat on a finished
  // progress bar with an unchanged caption for the length of a 600MB download. An installer
  // that looks hung gets cancelled, and a cancelled install lands the user in exactly the
  // half-finished state this gate exists to prevent.
  WizardForm.StatusLabel.Caption := Message;
  WizardForm.ProgressGauge.Style := npbstMarquee;
  WizardForm.Refresh();
end;

procedure InstallOllama();
var
  Code: Integer;
  Script: String;
begin
  if not WizardIsTaskSelected('ollama') then exit;
  Say('Installing Ollama and the embedding model. This downloads several hundred MB and can take a few minutes...');
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

procedure InitData();
var
  Code: Integer;
begin
  // The self-test opens the database read-only by design, so without this every fresh
  // install failed its own gate with "FAIL database" and told the marketer to go and start
  // a server. It also proves the data location is writable by whoever is running the
  // installer, which on Windows may be an administrator installing for somebody else.
  Say('Preparing the data directory...');
  Exec(ExpandConstant('{app}\{#AppExe}'), 'init', '', SW_HIDE, ewWaitUntilTerminated, Code);
end;

procedure ConfigureDesktop();
var
  Code: Integer;
begin
  if not WizardIsTaskSelected('desktop') then exit;
  // Never wire a product the self-test just rejected.
  if GSelfTestFailure <> '' then exit;
  Say('Connecting Claude Desktop...');
  Exec(ExpandConstant('{app}\{#AppExe}'), 'configure-desktop', '', SW_HIDE,
       ewWaitUntilTerminated, Code);
end;

procedure SelfTest();
var
  Output: String;
  Code: Integer;
begin
  Say('Checking that everything works...');
  Code := RunAndCapture(ExpandConstant('{app}\{#AppExe}'), 'health-check --wait 60', Output);
  if Code = 0 then exit;
  // Recorded, not raised - see the note on GSelfTestFailure. CurPageChanged below is what
  // the user actually reads, and ConfigureDesktop checks this before wiring anything.
  GSelfTestFailure := Output;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  // The actual gate. The Finished page is the only thing the user reads, so when the
  // self-test failed it must not say the product is ready.
  if (CurPageID = wpFinished) and (GSelfTestFailure <> '') then
  begin
    WizardForm.FinishedHeadingLabel.Caption := 'Installed, but not working';
    WizardForm.FinishedLabel.Caption :=
      'Campaign Intelligence was copied to this computer, but its self-test failed, so it ' +
      'is not ready to use.' + #13#10#13#10 + GSelfTestFailure + #13#10 +
      'Claude Desktop has NOT been connected, so nothing will try to use it yet.' +
      #13#10#13#10 + 'The files are in ' + ExpandConstant('{app}') + '. Fix what is named ' +
      'above and run this installer again.';
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // The self-test has to run BEFORE the Claude Desktop wiring: wiring first meant a
    // refused install still left the marketer's next session pointing at the server setup
    // had just condemned - the product review's own defect, with the installer's signature
    // on it. ConfigureDesktop is skipped entirely when the self-test failed.
    InstallOllama();
    InitData();
    SelfTest();
    ConfigureDesktop();
  end;
end;
