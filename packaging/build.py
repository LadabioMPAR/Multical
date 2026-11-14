"""
Builds the Windows program:  py packaging/build.py
(add --reaproveitar to skip step 1 and package the folder already in dist/Multical)

Steps:
1. PyInstaller makes the folder dist/Multical (Multical.exe plus its libraries);
2. the plugin folder and the user guide are copied next to the executable;
   when a code-signing certificate is configured (see below), every program
   file without a valid signature is signed;
3. the executable runs its self-check (`--autoteste`), including a plugin
   dropped in its folder; the build stops if anything fails;
4. dist/Multical-<version>-win64.zip is written (unzip and run, no installer);
5. if Inno Setup is installed, the installer dist/Multical-<version>-instalador.exe
   is compiled from packaging/installer.iss.

Needs: pip install pyinstaller PySide6-Essentials (and the package's own dependencies).

Windows may refuse to start an executable it has never seen ("Controle de
Aplicativo" / Smart App Control, error 4551) while it checks the file. The
self-check waits and tries a few times; if Windows keeps refusing, the build
stops and says so. That protection is not switched off or bypassed here.

Signing (optional). An unsigned program may be refused by Smart App Control on
the colleagues' machines too; a signature from a code-signing certificate
issued by a recognised authority is what Windows accepts. Nothing is signed
unless one of these is set, in the environment of the build (never in a file
of the repository):
  MULTICAL_CERT_THUMBPRINT   thumbprint of a certificate in the Windows store
                             (also for certificates kept on a token);
  MULTICAL_CERT_PFX          a .pfx file, with MULTICAL_CERT_PASSWORD.
Optional: MULTICAL_SIGNTOOL (path of signtool.exe, from the Windows SDK; found
by itself in the usual place) and MULTICAL_TIMESTAMP_URL (time-stamping server
of the certificate's authority; the signature stays valid after the
certificate expires). The installer and its uninstaller are signed by Inno
Setup with the same command.
"""
import os
import shutil
import subprocess
import sys
import time
import tomllib
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DIST = os.path.join(ROOT, "dist")
WORK = os.path.join(ROOT, "build", "pyinstaller")
APP = os.path.join(DIST, "Multical")

# Not needed by the program; leaving them out keeps the folder smaller.
EXCLUDED = ["torch", "tkinter", "pytest", "IPython", "notebook", "PyQt5", "PyQt6", "PySide2", "pandas"]

DEFAULT_TIMESTAMP = "http://timestamp.digicert.com"
SIGNED_KINDS = (".exe", ".dll", ".pyd")

BLOCKED_BY_WINDOWS = 4551      # ERROR_SYSTEM_INTEGRITY_POLICY_VIOLATION: application control held the file
START_ATTEMPTS, START_WAIT = 4, 30

SELF_TEST_PLUGIN = '''from multical.api import pretreatment

@pretreatment(id="plugin_do_autoteste", label="Plugin do autoteste")
def unchanged(X, wavelengths):
    return X, wavelengths
'''


def version():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def run_pyinstaller():
    import PyInstaller.__main__
    args = [
        os.path.join(HERE, "multical_entry.py"),
        "--name", "Multical", "--windowed", "--noconfirm", "--clean",
        "--distpath", DIST, "--workpath", WORK, "--specpath", WORK,
        "--paths", os.path.join(ROOT, "src"),
    ]
    for module in EXCLUDED:
        args += ["--exclude-module", module]
    PyInstaller.__main__.run(args)


def copy_extras():
    # The tutorial and one template per kind of method go along; nothing is active until the user copies a file.
    plugins = os.path.join(APP, "plugins")
    shutil.rmtree(plugins, ignore_errors=True)
    os.makedirs(plugins)
    source = os.path.join(ROOT, "plugins")
    shutil.copyfile(os.path.join(source, "LEIA-ME.md"), os.path.join(plugins, "LEIA-ME.md"))
    for folder in ("exemplos", "opcionais"):
        shutil.copytree(os.path.join(source, folder), os.path.join(plugins, folder),
                        ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(os.path.join(source, "exemplo_log10.py"), os.path.join(plugins, "exemplos", "pretratamento_log10.py"))
    guide = os.path.join(ROOT, "docs")
    if os.path.isfile(os.path.join(guide, "guia_do_usuario.html")):
        target = os.path.join(APP, "guia")
        shutil.rmtree(target, ignore_errors=True)
        os.makedirs(target)
        shutil.copyfile(os.path.join(guide, "guia_do_usuario.html"), os.path.join(target, "guia_do_usuario.html"))
        if os.path.isdir(os.path.join(guide, "img")):
            shutil.copytree(os.path.join(guide, "img"), os.path.join(target, "img"))


def signing_command():
    """The signtool command line without the files, or None when no certificate is configured."""
    thumbprint = os.environ.get("MULTICAL_CERT_THUMBPRINT", "").strip()
    pfx = os.environ.get("MULTICAL_CERT_PFX", "").strip()
    if not thumbprint and not pfx:
        return None
    tool = os.environ.get("MULTICAL_SIGNTOOL", "").strip() or shutil.which("signtool")
    if not tool:
        kits = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Windows Kits", "10", "bin")
        found = sorted(glob(os.path.join(kits, "*", "x64", "signtool.exe")))
        tool = found[-1] if found else None
    if not tool or not os.path.isfile(tool):
        raise SystemExit("Há um certificado configurado, mas o signtool.exe não foi encontrado. Instale o Windows SDK "
                         "(componente “Signing Tools”) ou indique o caminho em MULTICAL_SIGNTOOL.")
    command = [tool, "sign", "/fd", "sha256", "/td", "sha256",
               "/tr", os.environ.get("MULTICAL_TIMESTAMP_URL", "").strip() or DEFAULT_TIMESTAMP]
    if thumbprint:
        command += ["/sha1", thumbprint.replace(" ", "")]
    else:
        if not os.path.isfile(pfx):
            raise SystemExit(f"O certificado indicado em MULTICAL_CERT_PFX não existe: {pfx}")
        command += ["/f", pfx]
        if os.environ.get("MULTICAL_CERT_PASSWORD"):
            command += ["/p", os.environ["MULTICAL_CERT_PASSWORD"]]
    return command


def unsigned_files(folder):
    """Program files in `folder` without a valid signature (the libraries that come signed are left as they are)."""
    files = [os.path.join(top, name) for top, _, names in os.walk(folder)
             for name in names if name.lower().endswith(SIGNED_KINDS)]
    script = ("$input | ForEach-Object { $s = Get-AuthenticodeSignature -LiteralPath $_; "
              "if ($s.Status -ne 'Valid') { $_ } }")
    done = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                          input="\n".join(files), capture_output=True, text=True, check=True)
    return [line.strip() for line in done.stdout.splitlines() if line.strip()]


def sign_folder(command):
    """Sign every program file of the folder that is not signed yet."""
    files = unsigned_files(APP)
    for start in range(0, len(files), 50):      # signtool takes several files per call
        subprocess.run(command + files[start:start + 50], check=True, capture_output=True)
    left = unsigned_files(APP)
    if left:
        raise SystemExit(f"{len(left)} arquivo(s) continuam sem assinatura válida, por exemplo {left[0]}.")
    print(f"Assinados: {len(files)} arquivo(s) do programa.")


def self_test():
    """Run the packaged program's own checks, with one plugin in its folder. Returns the report text."""
    exe = os.path.join(APP, "Multical.exe")
    report = os.path.join(DIST, "autoteste.txt")
    plugin = os.path.join(APP, "plugins", "zz_autoteste.py")
    with open(plugin, "w", encoding="utf-8") as handle:
        handle.write(SELF_TEST_PLUGIN)
    environment = {k: v for k, v in os.environ.items() if k not in ("MULTICAL_PLUGINS", "PYTHONPATH")}
    try:
        if os.path.exists(report):
            os.remove(report)
        for attempt in range(1, START_ATTEMPTS + 1):
            started = time.perf_counter()
            try:
                done = subprocess.run([exe, "--autoteste", report], env=environment, cwd=DIST, timeout=600)
                break
            except OSError as error:
                if getattr(error, "winerror", None) != BLOCKED_BY_WINDOWS or attempt == START_ATTEMPTS:
                    if getattr(error, "winerror", None) == BLOCKED_BY_WINDOWS:
                        raise SystemExit(
                            "O Windows (Controle de Aplicativo) não deixou o executável novo rodar. Nada foi empacotado. "
                            "Tente de novo em alguns minutos com: py packaging/build.py --reaproveitar") from None
                    raise
                print(f"O Windows ainda está verificando o executável novo; nova tentativa em {START_WAIT} s "
                      f"({attempt} de {START_ATTEMPTS}).")
                time.sleep(START_WAIT)
        elapsed = time.perf_counter() - started
    finally:
        os.remove(plugin)
    text = open(report, encoding="utf-8").read() if os.path.exists(report) else "(o programa não escreveu o relatório)"
    print(text)
    print(f"Autoteste do executável: código de saída {done.returncode}, {elapsed:.1f} s")
    if done.returncode != 0 or "plugin_do_autoteste" not in text:
        raise SystemExit("O executável não passou no autoteste; nada foi empacotado.")
    return text


def folder_size(path):
    return sum(os.path.getsize(os.path.join(folder, name)) for folder, _, names in os.walk(path) for name in names)


def find_inno_setup():
    candidates = [shutil.which("iscc")]
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")):
        if base:
            candidates.append(os.path.join(base, "Inno Setup 6", "ISCC.exe"))
    return next((c for c in candidates if c and os.path.isfile(c)), None)


def main():
    number = version()
    if "--reaproveitar" in sys.argv[1:]:
        if not os.path.isfile(os.path.join(APP, "Multical.exe")):
            raise SystemExit("Não há programa em dist/Multical para reaproveitar; rode sem --reaproveitar.")
    else:
        run_pyinstaller()
    copy_extras()
    signer = signing_command()
    if signer:
        sign_folder(signer)
    else:
        print("Nenhum certificado configurado: o programa não foi assinado.")
    self_test()
    print(f"Pasta do programa: {APP} ({folder_size(APP) / 1e6:.0f} MB)")
    archive = shutil.make_archive(os.path.join(DIST, f"Multical-{number}-win64"), "zip", DIST, "Multical")
    print(f"Arquivo zip: {archive} ({os.path.getsize(archive) / 1e6:.0f} MB)")
    compiler = find_inno_setup()
    if compiler:
        options = [f"/DAppVersion={number}"]
        if signer:      # Inno Setup signs the installer and the uninstaller it writes
            options += ["/DSignInstaller", "/Smultical=" + " ".join(f"$q{part}$q" for part in signer) + " $f"]
        subprocess.run([compiler, *options, os.path.join(HERE, "installer.iss")], check=True)
        print(f"Instalador: {os.path.join(DIST, f'Multical-{number}-instalador.exe')}")
    else:
        print("Inno Setup não encontrado: o instalador não foi gerado (o zip acima já pode ser distribuído).")


if __name__ == "__main__":
    sys.exit(main())
