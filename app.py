import os
import pathlib
import sys

import matplotlib
matplotlib.use('QT5Agg')
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QSlider

import fitting
from mainapp import Ui_MainWindow

BG = (242/255, 243/255, 245/255)
OVERLAPS = list(range(0, 26, 5))
# combo-box index -> parameter name
AXIS_NAMES = {0: "number of chunks", 1: "polynomial order", 2: "overlapping percentage"}


def poly_latex(coeffs, digits=2):
    """LaTeX for a polynomial given highest-degree-first coefficients."""
    deg = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        p = deg - i
        if round(c, digits) == 0:
            continue
        mag = f"{abs(c):.{digits}f}"
        var = "" if p == 0 else ("x" if p == 1 else f"x^{{{p}}}")
        body = mag + (" " + var if var else "")
        sign = "-" if c < 0 else "+"
        terms.append((sign, body))
    if not terms:
        return "0"
    first_sign, first = terms[0]
    out = ("-" if first_sign == "-" else "") + first
    for sign, body in terms[1:]:
        out += f" {sign} {body}"
    return out


class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, dpi=120):
        self.fig = Figure(dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.axes.set_facecolor(BG)
        self.fig.set_facecolor(BG)


class ErrorMapWorker(QtCore.QObject):
    """Computes the three error maps off the GUI thread; never touches widgets."""
    finished = QtCore.pyqtSignal(object)

    def __init__(self, x, y, max_order, max_chunks):
        super().__init__()
        self.x, self.y = x, y
        self.max_order, self.max_chunks = max_order, max_chunks
        self.stop = False

    def run(self):
        x, y = self.x, self.y
        n = len(x)
        orders = list(range(1, self.max_order + 1))
        chunks = list(range(1, min(self.max_chunks, n) + 1))
        result = {"orders": orders, "chunks": chunks, "overlaps": OVERLAPS}
        jobs = [
            ("order_chunks", lambda: fitting.error_map(x, y, orders, chunks, [0])),
            ("chunks_overlap", lambda: fitting.error_map(x, y, [5], chunks, OVERLAPS)),
            ("order_overlap", lambda: fitting.error_map(x, y, orders, [5], OVERLAPS)),
        ]
        for key, job in jobs:
            if self.stop:
                self.finished.emit(None)
                return
            try:
                result[key] = job()
            except ValueError:
                # e.g. a slider at 1 leaves an axis with a single value
                result[key] = None
        self.finished.emit(None if self.stop else result)


class mainApp(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        for slider in (self.ui.SliderOfOrder, self.ui.SliderOfChunks):
            slider.setMinimum(0)
            slider.setMaximum(10)
            slider.setTickInterval(1)
            slider.setTickPosition(QSlider.TicksBelow)

        self.mainSignalDataPlotter = self.ui.mainSignal.plot([], [], pen="k")
        self.verticalLinePlotter = self.ui.mainSignal.plot([], [], pen="g")
        self.extraPolationPlotter = self.ui.mainSignal.plot([], [], pen="r")

        self.chunkplotter = self.ui.mainSignal
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

        self.xAxisValuesOfOriginalSignal = np.array([])
        self.yAxisValuesOfOriginalSignal = np.array([])

        self.ui.showErrorMapCheckBox.stateChanged.connect(self.showHideErrorMap)
        self.ui.extrapolationCheckBox.stateChanged.connect(self.extraPolationToggle)
        self.ui.generateErrorMapBtn.clicked.connect(self.errorMapStart)
        self.ui.cancelBtn.clicked.connect(self.errorMapCancel)
        self.ui.xAxesComboBox.currentIndexChanged['int'].connect(self.showErrorMap)
        self.ui.yAxesComboBox.currentIndexChanged['int'].connect(self.showErrorMap)
        self.errorMaps = None
        self.errorMapThread = None
        self.errorMapWorker = None
        divider = make_axes_locatable(self.errorMapCanv.axes)
        self.cax = divider.append_axes('right', size='5%', pad=0.05)
        self.cax.set_facecolor(BG)

    # ---------------------------------------------------------------- error map
    def errorMapStart(self):
        if len(self.xAxisValuesOfOriginalSignal) == 0:
            return
        if self.errorMapThread is not None and self.errorMapThread.isRunning():
            return
        self.errorMapThread = QtCore.QThread(self)
        self.errorMapWorker = ErrorMapWorker(
            self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal,
            self.ui.orderOfFittingPolynomialErrorMapSlider.value(),
            self.ui.numberOfChunksErrorMapSlider.value())
        self.errorMapWorker.moveToThread(self.errorMapThread)
        self.errorMapThread.started.connect(self.errorMapWorker.run)
        # Queued back to the GUI thread: all drawing happens in errorMapDone.
        self.errorMapWorker.finished.connect(self.errorMapDone)
        self.errorMapWorker.finished.connect(self.errorMapThread.quit)
        self.errorMapThread.start()

    def errorMapCancel(self):
        if self.errorMapWorker is not None:
            self.errorMapWorker.stop = True

    def errorMapDone(self, result):
        if result is None:
            return
        self.errorMaps = result
        self.showErrorMap()

    def showHideErrorMap(self, state):
        self.errorMapCanv.setVisible(state == QtCore.Qt.Checked)

    def _map_for_axes(self, xi, yi):
        """Return (image rows=y param, cols=x param, x values, y values) or None."""
        m = self.errorMaps
        values = {0: m["chunks"], 1: m["orders"], 2: m["overlaps"]}
        # stored maps: axis0 / axis1 parameter indices
        stored = {(1, 0): "order_chunks", (0, 2): "chunks_overlap", (1, 2): "order_overlap"}
        if (yi, xi) in stored:
            data = m[stored[(yi, xi)]]
        elif (xi, yi) in stored:
            data = m[stored[(xi, yi)]]
            data = None if data is None else data.T
        else:
            return None
        if data is None:
            return None
        return data, values[xi], values[yi]

    def showErrorMap(self):
        if self.errorMaps is None:
            return
        self.errorMapCanv.axes.clear()
        self.cax.clear()
        xi = self.ui.xAxesComboBox.currentIndex()
        yi = self.ui.yAxesComboBox.currentIndex()
        found = self._map_for_axes(xi, yi)
        if found is not None:
            data, xs, ys = found
            peak = np.nanmax(data)
            normalized = data / peak * 100 if peak > 0 else data  # % of worst cell
            image = self.errorMapCanv.axes.imshow(
                normalized, cmap="inferno", origin="lower", aspect="auto",
                extent=[xs[0] - 0.5, xs[-1] + 0.5, ys[0] - 0.5, ys[-1] + 0.5]
                if len(xs) > 1 and len(ys) > 1 else None)
            self.errorMapCanv.axes.set_xlabel(AXIS_NAMES[xi])
            self.errorMapCanv.axes.set_ylabel(AXIS_NAMES[yi])
            self.errorMapCanv.fig.colorbar(image, cax=self.cax, orientation='vertical')
        self.errorMapCanv.fig.tight_layout()
        self.errorMapCanv.draw()

    # ------------------------------------------------------------ extrapolation
    def extraPolationToggle(self, state):
        if state == QtCore.Qt.Checked:
            self.ui.SliderOfOrder.valueChanged.disconnect(self.chunks)
            self.ui.SliderOfChunks.valueChanged.disconnect(self.chunks)
            self.chunkplotter.clear()
            self.ui.chunksLabel.setText("extrapolation")
            self.ui.SliderOfChunks.valueChanged.connect(self.updateExtraPolation)
            self.ui.SliderOfOrder.valueChanged.connect(self.updateExtraPolation)
            self.mainSignalDataPlotter = self.ui.mainSignal.plot(
                self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal, pen="k")
            self.verticalLinePlotter = self.ui.mainSignal.plot([], [], pen="g")
            self.extraPolationPlotter = self.ui.mainSignal.plot([], [], pen="r")
            self.updateExtraPolation()
        else:
            self.ui.SliderOfOrder.valueChanged.disconnect(self.updateExtraPolation)
            self.ui.SliderOfChunks.valueChanged.disconnect(self.updateExtraPolation)
            self.ui.chunksLabel.setText("Chunks")
            self.ui.SliderOfChunks.valueChanged.connect(self.chunks)
            self.ui.SliderOfOrder.valueChanged.connect(self.chunks)
            self.verticalLinePlotter.setData([], [])
            self.extraPolationPlotter.setData([], [])
            self.chunks()

    def updateExtraPolation(self):
        x, y = self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal
        fraction = self.ui.SliderOfChunks.value() / self.ui.SliderOfChunks.maximum()
        order = self.ui.SliderOfOrder.value()
        if len(x) == 0 or fraction <= 0 or len(x) < order + 1:
            self.verticalLinePlotter.setData([], [])
            self.extraPolationPlotter.setData([], [])
            return
        _, y_pred = fitting.extrapolate(x, y, fraction, order)
        n_fit = min(len(x), max(order + 1, int(round(fraction * len(x)))))
        x_line = x[n_fit - 1]
        self.verticalLinePlotter.setData([x_line, x_line], [y.min(), y.max()])
        self.extraPolationPlotter.setData(x, y_pred)

    # ---------------------------------------------------------------- file / fit
    def read_file(self):
        path = QFileDialog.getOpenFileName()[0]
        if not path:
            return
        self.FileName = os.path.basename(path)
        if pathlib.Path(path).suffix == ".csv":
            self.data = np.genfromtxt(path, delimiter=',')
            self.xAxisValuesOfOriginalSignal = self.data[:, 0][:1000]
            self.yAxisValuesOfOriginalSignal = self.data[:, 1][:1000]
            self.mainSignalDataPlotter.setData(self.xAxisValuesOfOriginalSignal,
                                               self.yAxisValuesOfOriginalSignal)
            self.ui.mainSignal.setLimits(yMin=self.yAxisValuesOfOriginalSignal.min(),
                                         yMax=self.yAxisValuesOfOriginalSignal.max())

    def chunks(self):
        x, y = self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal
        n_chunks = self.ui.SliderOfChunks.value()
        order = self.ui.SliderOfOrder.value()
        self.chunkplotter.clear()
        self.chunkplotter.plot(x, y, pen="k")
        if n_chunks <= 0 or len(x) < n_chunks:
            return

        rows = []
        for i, fit in enumerate(fitting.fit_chunks(x, y, n_chunks, order)):
            xs = np.linspace(fit.x_min, fit.x_max)
            self.chunkplotter.plot(xs, fit.predict(xs), symbol='o')
            rows.append([f'eq of chunk {i + 1}', f'${poly_latex(fit.coeffs)}$',
                         round(fit.rmse, 5)])

        self.latexWriter.axes.clear()
        if order > 0:  # plot the table only if the slider has a value
            columns = ['Chunk', 'equation', 'RMSE']
            table = self.latexWriter.axes.table(cellText=rows, colLabels=columns, loc='center')
            table.auto_set_column_width(col=list(range(len(columns))))
            table.scale(1, 1.5)
            table.auto_set_font_size(True)
        self.latexWriter.fig.tight_layout()
        self.latexWriter.axes.set_axis_off()
        self.latexWriter.draw()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = mainApp()
    main.show()
    sys.exit(app.exec_())
