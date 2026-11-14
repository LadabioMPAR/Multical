"""
The optional code signing of the Windows build (packaging/build.py), without a
certificate: what is signed and with which command. Nothing is signed here.
"""
import importlib.util
import os
import tempfile
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location("multical_build", os.path.join(REPO, "packaging", "build.py"))
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)

NO_CERTIFICATE = {k: v for k, v in os.environ.items() if not k.startswith("MULTICAL_CERT")}


class Signing(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tool = os.path.join(self._tmp.name, "signtool.exe")
        open(self.tool, "wb").close()

    def test_nothing_is_signed_without_a_certificate(self):
        with mock.patch.dict(os.environ, NO_CERTIFICATE, clear=True):
            self.assertIsNone(build.signing_command())

    def test_certificate_from_the_windows_store(self):
        env = {**NO_CERTIFICATE, "MULTICAL_CERT_THUMBPRINT": "ab cd 12", "MULTICAL_SIGNTOOL": self.tool}
        with mock.patch.dict(os.environ, env, clear=True):
            command = build.signing_command()
        self.assertEqual(command[:2], [self.tool, "sign"])
        self.assertEqual(command[command.index("/sha1") + 1], "abcd12")
        self.assertEqual(command[command.index("/fd") + 1], "sha256")
        self.assertEqual(command[command.index("/tr") + 1], build.DEFAULT_TIMESTAMP)
        self.assertNotIn("/f", command)

    def test_certificate_file_with_password_and_own_timestamp_server(self):
        pfx = os.path.join(self._tmp.name, "certificado.pfx")
        open(pfx, "wb").close()
        env = {**NO_CERTIFICATE, "MULTICAL_CERT_PFX": pfx, "MULTICAL_CERT_PASSWORD": "segredo",
               "MULTICAL_SIGNTOOL": self.tool, "MULTICAL_TIMESTAMP_URL": "http://carimbo.exemplo"}
        with mock.patch.dict(os.environ, env, clear=True):
            command = build.signing_command()
        self.assertEqual((command[command.index("/f") + 1], command[command.index("/p") + 1]), (pfx, "segredo"))
        self.assertEqual(command[command.index("/tr") + 1], "http://carimbo.exemplo")

    def test_missing_pieces_are_explained(self):
        missing = os.path.join(self._tmp.name, "nao_existe.pfx")
        with mock.patch.dict(os.environ, {**NO_CERTIFICATE, "MULTICAL_CERT_PFX": missing, "MULTICAL_SIGNTOOL": self.tool},
                             clear=True):
            with self.assertRaisesRegex(SystemExit, "não existe"):
                build.signing_command()
        with mock.patch.dict(os.environ, {**NO_CERTIFICATE, "MULTICAL_CERT_THUMBPRINT": "ab",
                                          "MULTICAL_SIGNTOOL": os.path.join(self._tmp.name, "outro.exe")}, clear=True):
            with self.assertRaisesRegex(SystemExit, "signtool.exe não foi encontrado"):
                build.signing_command()

    def test_only_program_files_without_a_valid_signature_are_signed(self):
        folder = os.path.join(self._tmp.name, "Multical")
        os.makedirs(os.path.join(folder, "_internal"))
        names = ["Multical.exe", os.path.join("_internal", "modulo.pyd"), os.path.join("_internal", "biblioteca.dll"),
                 os.path.join("_internal", "dados.json"), os.path.join("_internal", "codigo.py")]
        for name in names:
            with open(os.path.join(folder, name), "wb") as handle:
                handle.write(b"MZ")
        found = sorted(os.path.relpath(p, folder) for p in build.unsigned_files(folder))
        self.assertEqual(found, sorted(names[:3]))              # json and py are not program files


class InstallerScript(unittest.TestCase):
    def test_installer_signs_only_when_asked(self):
        with open(os.path.join(REPO, "packaging", "installer.iss"), encoding="utf-8") as handle:
            text = handle.read()
        block = text.split("#ifdef SignInstaller")[1].split("#endif")[0]
        self.assertIn("SignTool=multical", block)
        self.assertIn("SignedUninstaller=yes", block)


if __name__ == "__main__":
    unittest.main()
