import os
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QSlider
from sympy import S, printing, symbols

import fitting
from fitting_cli import load_xy
from mainapp import Ui_MainWindow

matplotlib.use('QT5Agg')

# Axis variables, indexed like the X/Y combo boxes in mainapp.ui.
CHUNKS, ORDER, OVERLAP = 0, 1, 2
AXIS_NAMES = {
    CHUNKS: "number of chunks",
    ORDER: "polynomial order",
    OVERLAP: "overlapping percentage",
}
OVERLAP_STEPS = list(range(0, 26, 5))
OVERLAP_MAX = OVERLAP_STEPS[-1]
# Order / chunk count used by the maps that do not vary them (same as before the rewrite).
FIXED_ORDER = 5
FIXED_CHUNKS = 5
BACKGROUND = (242 / 255, 243 / 255, 245 / 255)


class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, dpi=120):
        self.fig, self.axes = plt.subplots()
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.axes.set_facecolor(BACKGROUND)
        self.fig.set_facecolor(BACKGROUND)


def normalize_percent(error_array):
    """Return a copy scaled so the maximum is 100. Never modifies the input."""
    peak = error_array.max()
    if peak <= 0:
        return np.zeros_like(error_array, dtype=float)
    return error_array / peak * 100.0


class ErrorMapWorker(QtCore.QThread):
    """Computes the three error maps off the GUI thread. Emits only plain data; the GUI draws."""

    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, x, y, max_order, max_chunks, parent=None):
        super().__init__(parent)
        self._x = x
        self._y = y
        self._max_order = max_order
        self._max_chunks = max_chunks

    def run(self):
        orders = list(range(1, self._max_order + 1))
        chunks = list(range(1, self._max_chunks + 1))
        # (axis0 variable, axis1 variable) -> un-normalized RMSE array, as returned by fitting.error_map
        jobs = [
            ((ORDER, CHUNKS), (orders, chunks, [0])),
            ((CHUNKS, OVERLAP), ([FIXED_ORDER], chunks, OVERLAP_STEPS)),
            ((ORDER, OVERLAP), (orders, [FIXED_CHUNKS], OVERLAP_STEPS)),
        ]
        try:
            maps = {}
            for key, (order_vals, chunk_vals, overlap_vals) in jobs:
                if self.isInterruptionRequested():
                    self.cancelled.emit()
                    return
                raw = fitting.error_map(self._x, self._y, order_vals, chunk_vals, overlap_vals)
                norm = normalize_percent(raw)
                # Both orientations are built from the normalized data.
                maps[key] = norm
                maps[key[::-1]] = np.ascontiguousarray(norm.T)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.failed.emit(f"error map failed: {exc}")
            return
        self.done.emit(maps)


class mainApp(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(mainApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.SliderOfOrder.setMinimum(0)
        self.ui.SliderOfOrder.setMaximum(10)
        self.ui.SliderOfOrder.setTickInterval(1)
        self.ui.SliderOfOrder.setTickPosition(QSlider.TicksBelow)

        self.ui.SliderOfChunks.setMinimum(0)
        self.ui.SliderOfChunks.setMaximum(10)
        self.ui.SliderOfChunks.setTickInterval(1)
        self.ui.SliderOfChunks.setTickPosition(QSlider.TicksBelow)

        self.chunkplotter = self.ui.mainSignal
        self.verticalLinePlotter = self.ui.mainSignal.plot([], [], pen="g")
        self.extraPolationPlotter = self.ui.mainSignal.plot([], [], pen="r")

        self.xData = None
        self.yData = None

        self.ui.actionopen.triggered.connect(lambda: self.read_file())
        self.ui.SliderOfOrder.valueChanged.connect(self.chunks)
        self.ui.SliderOfChunks.valueChanged.connect(self.chunks)

        self.errorMapCanv = MatplotlibCanvas(self)
        self.errorMapCanv.setMaximumWidth(427)
        self.errorMapCanv.setMaximumHeight(260)
        self.ui.errorMapLayout.addWidget(self.errorMapCanv)
        self.errorMapCanv.setVisible(False)

        self.latexWriter = MatplotlibCanvas(self)
        self.latexWriter.setMaximumHeight(260)
        self.latexWriter.axes.set_axis_off()
        self.ui.latexLayout.addWidget(self.latexWriter)

        self.ui.showErrorMapCheckBox.stateChanged.connect(self.showHideErrorMap)
        self.ui.extrapolationCheckBox.stateChanged.connect(self.extraPolationToggle)
        self.ui.generateErrorMapBtn.clicked.connect(self.errorMapStart)
        self.ui.cancelBtn.clicked.connect(self.errorMapCancel)
        self.ui.xAxesComboBox.currentIndexChanged['int'].connect(self.showErrorMap)
        self.ui.yAxesComboBox.currentIndexChanged['int'].connect(self.showErrorMap)

        self.errorMapGenerated = False
        self._errorMapMaps = {}
        self._errorMapBounds = {}
        self._errorMapWorker = None

        divider = make_axes_locatable(self.errorMapCanv.axes)
        self.cax = divider.append_axes('right', size='5%', pad=0.05)
        self.cax.set_facecolor(BACKGROUND)

    def _report(self, message):
        self.ui.statusbar.showMessage(message, 5000)

    def _hasData(self):
        return self.xData is not None and len(self.xData) > 0

    def closeEvent(self, event):
        worker = self._errorMapWorker
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait()
        super().closeEvent(event)

    # ---- error map -------------------------------------------------------

    def errorMapStart(self):
        if self._errorMapWorker is not None and self._errorMapWorker.isRunning():
            return
        if not self._hasData():
            self._report("load a CSV file first")
            return
        max_order = self.ui.orderOfFittingPolynomialErrorMapSlider.value()
        max_chunks = self.ui.numberOfChunksErrorMapSlider.value()
        if max_order < 2 or max_chunks < 2:
            self._report("error map needs order and chunk sliders at 2 or more")
            return

        self.errorMapGenerated = False
        self._errorMapBounds = {CHUNKS: max_chunks, ORDER: max_order, OVERLAP: OVERLAP_MAX}
        self._report("generating error map...")
        worker = ErrorMapWorker(self.xData.copy(), self.yData.copy(), max_order, max_chunks)
        worker.done.connect(self._onErrorMapReady)
        worker.failed.connect(self._report)
        worker.cancelled.connect(lambda: self._report("error map cancelled"))
        self._errorMapWorker = worker
        worker.start()

    def errorMapCancel(self):
        if self._errorMapWorker is not None and self._errorMapWorker.isRunning():
            self._errorMapWorker.requestInterruption()

    def _onErrorMapReady(self, maps):
        self._errorMapMaps = maps
        self.errorMapGenerated = True
        self._report("error map ready")
        self.showErrorMap()

    def showHideErrorMap(self, state):
        if (state == QtCore.Qt.Checked):
            self.errorMapCanv.setVisible(True)
        else:
            self.errorMapCanv.setVisible(False)

    def showErrorMap(self):
        if not self.errorMapGenerated:
            return

        x = self.ui.xAxesComboBox.currentIndex()
        y = self.ui.yAxesComboBox.currentIndex()
        if x == y:
            self._report("choose two different axis variables")
            return

        self.errorMapCanv.axes.clear()
        self.cax.clear()

        # Maps are stored with rows = first key variable; flip so the larger values sit at the top.
        display = np.flipud(self._errorMapMaps[(y, x)])
        extent = [0, self._errorMapBounds[x], 0, self._errorMapBounds[y]]
        image = self.errorMapCanv.axes.imshow(display, cmap="inferno", extent=extent)
        self.errorMapCanv.axes.set_xlabel(AXIS_NAMES[x])
        self.errorMapCanv.axes.set_ylabel(AXIS_NAMES[y])

        self.errorMapCanv.fig.colorbar(image, cax=self.cax, orientation='vertical')
        self.errorMapCanv.fig.tight_layout()
        self.errorMapCanv.draw()

    # ---- extrapolation ----------------------------------------------------

    def extraPolationToggle(self, state):
        if (state == QtCore.Qt.Checked):
            self.ui.SliderOfOrder.valueChanged.disconnect(self.chunks)
            self.ui.SliderOfChunks.valueChanged.disconnect(self.chunks)
            self.ui.chunksLabel.setText("extrapolation")
            self.ui.SliderOfChunks.valueChanged.connect(self.updateExtraPolation)
            self.ui.SliderOfOrder.valueChanged.connect(self.updateExtraPolation)
            self._showExtrapolationBase()
            self.updateExtraPolation()
        else:
            self.ui.SliderOfOrder.valueChanged.disconnect(self.updateExtraPolation)
            self.ui.SliderOfChunks.valueChanged.disconnect(self.updateExtraPolation)
            self.ui.chunksLabel.setText("Chunks")
            self.ui.SliderOfChunks.valueChanged.connect(self.chunks)
            self.ui.SliderOfOrder.valueChanged.connect(self.chunks)
            self.chunks()

    def _showExtrapolationBase(self):
        self.chunkplotter.clear()
        if self._hasData():
            self.chunkplotter.plot(self.xData, self.yData, pen="k")
        self.verticalLinePlotter = self.ui.mainSignal.plot([], [], pen="g")
        self.extraPolationPlotter = self.ui.mainSignal.plot([], [], pen="r")

    def updateExtraPolation(self):
        self.verticalLinePlotter.setData([], [])
        self.extraPolationPlotter.setData([], [])
        if not self._hasData():
            return
        slider = self.ui.SliderOfChunks.value()
        if slider == 0:
            return

        fraction = slider / 10
        order = self.ui.SliderOfOrder.value()
        try:
            _, y_pred = fitting.extrapolate(self.xData, self.yData, fraction, order)
        except ValueError as exc:
            self._report(f"extrapolation failed: {exc}")
            return

        n = len(self.xData)
        last_fitted = min(n, max(order + 1, int(np.ceil(fraction * n)))) - 1
        cut_x = self.xData[last_fitted]
        self.verticalLinePlotter.setData([cut_x, cut_x], [self.yData.min(), self.yData.max()])
        self.extraPolationPlotter.setData(self.xData, y_pred)

    # ---- data and chunk fits ----------------------------------------------

    def read_file(self):
        path, _ = QFileDialog.getOpenFileName()
        if not path:
            return
        if pathlib.Path(path).suffix.lower() != ".csv":
            self._report("only .csv files are supported")
            return
        try:
            x, y = load_xy(path)
        except (OSError, ValueError) as exc:
            self._report(f"could not read {os.path.basename(path)}: {exc}")
            return

        self.xData = x
        self.yData = y
        self.errorMapGenerated = False
        self.ui.mainSignal.setLimits(yMin=float(y.min()), yMax=float(y.max()))
        if self.ui.extrapolationCheckBox.isChecked():
            self._showExtrapolationBase()
            self.updateExtraPolation()
        else:
            self.chunks()

    def chunks(self):
        self.chunkplotter.clear()
        self.latexWriter.axes.clear()
        if not self._hasData():
            self.latexWriter.draw()
            return

        x, y = self.xData, self.yData
        self.chunkplotter.plot(x, y, pen="k")
        n_chunks = self.ui.SliderOfChunks.value()
        order = self.ui.SliderOfOrder.value()

        if n_chunks > 0:
            try:
                fits = fitting.fit_chunks(x, y, n_chunks, order)
            except ValueError as exc:
                self._report(f"fit failed: {exc}")
                return
            table_data = self._chunk_table_data(fits, y)
            for fit in fits:
                xs = np.linspace(fit.x_min, fit.x_max)
                self.chunkplotter.plot(xs, fit.predict(xs), symbol='o')

            if order > 0:
                frame = pd.DataFrame(table_data)
                table = self.latexWriter.axes.table(cellText=frame.values, colLabels=frame.columns, loc='center')
                table.auto_set_column_width(col=list(range(len(frame.columns))))
                table.scale(1, 1.5)
                table.auto_set_font_size(True)

        self.latexWriter.fig.tight_layout()
        self.latexWriter.axes.set_axis_off()
        self.latexWriter.draw()

    @staticmethod
    def _chunk_table_data(fits, y):
        """error% = chunk RMSE as a percentage of the signal's peak-to-peak range."""
        span = np.ptp(y)
        x_sym = symbols("x")
        table = {'Chunk': [], 'equation': [], 'error%': []}
        for idx, fit in enumerate(fits, start=1):
            poly = sum(S("{:6.2f}".format(c)) * x_sym ** p for p, c in enumerate(fit.coeffs[::-1]))
            table['Chunk'].append(f'eq of chunk {idx}')
            table['equation'].append(f'${printing.latex(poly)}$')
            table['error%'].append(round(fit.rmse / span * 100, 3) if span > 0 else float('nan'))
        return table


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = mainApp()
    main.show()
    sys.exit(app.exec_())
