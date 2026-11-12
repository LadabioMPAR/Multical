"""Runs the calibration outside the window's thread, so the interface keeps responding."""
import traceback

from PySide6.QtCore import QThread, Signal

from multical import api

from .state import load_spectra


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


class PredictionWorker(QThread):
    """Reads the spectra files and estimates every sample with a model."""
    progress = Signal(float, str)
    succeeded = Signal(object)      # api.Prediction
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, model, sources, every=1, parent=None):
        super().__init__(parent)
        self._model, self._sources, self._every = model, list(sources), every
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            parts = []
            for number, path in enumerate(self._sources, start=1):
                if self._cancel:
                    raise api.CalibrationCancelled()
                self.progress.emit(0.0, f"Lendo o arquivo {number} de {len(self._sources)}…")
                parts.append(load_spectra(path, self._every, taken=[str(p.groups[0]) for p in parts]))
            prediction = api.predict_report(
                self._model, api.Dataset.concat(parts), progress=self.progress.emit, cancel=lambda: self._cancel)
        except api.CalibrationCancelled:
            self.cancelled.emit()
        except (ValueError, OSError) as error:
            self.failed.emit(str(error))
        except Exception as error:
            self.failed.emit(f"Erro inesperado: {error}\n\n{traceback.format_exc(limit=4)}")
        else:
            self.succeeded.emit(prediction)
