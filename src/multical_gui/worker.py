"""Runs the calibration outside the window's thread, so the interface keeps responding."""
import traceback

from PySide6.QtCore import QThread, Signal

from multical import api


class CalibrationWorker(QThread):
    progress = Signal(float, str)
    succeeded = Signal(object)      # CalibrationResult
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, dataset, config, parent=None):
        super().__init__(parent)
        self._dataset, self._config = dataset, config
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            result = api.run_calibration(
                self._dataset, self._config, progress=self.progress.emit, cancel=lambda: self._cancel)
        except api.CalibrationCancelled:
            self.cancelled.emit()
        except ValueError as error:          # messages written for the user
            self.failed.emit(str(error))
        except Exception as error:           # anything else: keep the detail for whoever maintains the program
            self.failed.emit(f"Erro inesperado: {error}\n\n{traceback.format_exc(limit=4)}")
        else:
            self.succeeded.emit(result)
