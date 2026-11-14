; Windows installer for Multical (Inno Setup 6).
; Compiled by packaging/build.py when Inno Setup is installed:
;   ISCC /DAppVersion=0.1.0 packaging\installer.iss
; It installs for the current user only (no administrator password), so the
; "plugins" folder next to the program stays writable.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6C7B1E0A-5B1F-4C39-9B55-4D2A0B7E91C3}
AppName=Multical
AppVersion={#AppVersion}
AppPublisher=LadabioMPAR
DefaultDirName={localappdata}\Programs\Multical
DefaultGroupName=Multical
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=Multical-{#AppVersion}-instalador
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\Multical.exe
#ifdef SignInstaller
; build.py defines SignInstaller and passes the signing command as /Smultical=... when a certificate is configured
SignTool=multical
SignedUninstaller=yes
#endif

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\Multical\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Multical"; Filename: "{app}\Multical.exe"
Name: "{group}\Guia do usuário"; Filename: "{app}\guia\guia_do_usuario.html"
Name: "{autodesktop}\Multical"; Filename: "{app}\Multical.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Multical.exe"; Description: "{cm:LaunchProgram,Multical}"; Flags: nowait postinstall skipifsilent
