import os
import pathlib
import sys
import threading

import matplotlib

matplotlib.use('Qt5Agg')

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QSlider
from sympy import S, printing, symbols

import fitting
from mainapp import Ui_MainWindow

BACKGROUND = (242 / 255, 243 / 255, 245 / 255)
ROW_LIMIT = 1000
CHUNK_PLOT_POINTS = 50
EXTRAPOLATION_MIN_FRACTION = 1e-6
OVERLAPS = list(range(0, 26, 5))
AXIS_NAMES = ('number of chunks', 'polynomial order', 'overlapping percentage')
# map key -> (row axis, column axis); axis ids: 0 chunks, 1 order, 2 overlap
MAP_AXES = {
    'order_chunks': (1, 0),
    'chunks_overlap': (0, 2),
    'order_overlap': (1, 2),
}


def compute_error_maps(x, y, max_order, max_chunks, fixed_order, is_cancelled=lambda: False):
    """Raw RMSE maps for the three axis pairs. Returns None if cancelled between maps."""
    if max_order < 2 or max_chunks < 2:
        raise ValueError("max_order and max_chunks must be >= 2")
    orders = list(range(1, max_order + 1))
    chunks = list(range(1, min(max_chunks, len(x)) + 1))
    fixed_chunks = min(5, len(x))
    jobs = {
        'order_chunks': lambda: fitting.error_map(x, y, orders, chunks, [0]),
        'chunks_overlap': lambda: fitting.error_map(x, y, [fixed_order], chunks, OVERLAPS),
        'order_overlap': lambda: fitting.error_map(x, y, orders, [fixed_chunks], OVERLAPS),
    }
    result = {'orders': orders, 'chunks': chunks, 'overlaps': OVERLAPS}
    for key, job in jobs.items():
        if is_cancelled():
            return None
        result[key] = job()
    return result


def normalize_to_percent(m):
    out = np.array(m, dtype=float)
    if np.isfinite(out).any() and np.nanmax(out) > 0:
        out = out / np.nanmax(out) * 100
    return out


def cell_extent(values):
    step = values[1] - values[0] if len(values) > 1 else 1
    return float(values[0] - step / 2), float(values[-1] + step / 2)


def latex_equation(coeffs):
    x = symbols('x')
    poly = sum(S(f"{c:6.2f}") * x ** i for i, c in enumerate(coeffs[::-1]))
    return printing.latex(poly)


class ErrorMapWorker(QtCore.QThread):
    """Computes numbers only; the GUI thread draws the result."""
    resultReady = QtCore.pyqtSignal(object)

    def __init__(self, x, y, max_order, max_chunks, fixed_order, parent=None):
        super().__init__(parent)
        self._args = (x, y, max_order, max_chunks, fixed_order)
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        self.resultReady.emit(compute_error_maps(*self._args, is_cancelled=self._cancel.is_set))


class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(facecolor=BACKGROUND)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.axes.set_facecolor(BACKGROUND)


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
        self.FileName = ""

        self.errorMapWorker = None
        self.errorMapGenerated = False
        self.errorMaps = {}
        self.errorMapAxes = {}

        self.ui.showErrorMapCheckBox.stateChanged.connect(self.showHideErrorMap)
        self.ui.extrapolationCheckBox.stateChanged.connect(self.extraPolationToggle)
        self.ui.generateErrorMapBtn.clicked.connect(self.errorMapStart)
        self.ui.cancelBtn.clicked.connect(self.errorMapCancel)
        self.ui.xAxesComboBox.currentIndexChanged.connect(lambda _index: self.showErrorMap())
        self.ui.yAxesComboBox.currentIndexChanged.connect(lambda _index: self.showErrorMap())
        divider = make_axes_locatable(self.errorMapCanv.axes)
        self.cax = divider.append_axes('right', size='5%', pad=0.05)
        self.cax.set_facecolor(BACKGROUND)

    def closeEvent(self, event):
        if self.errorMapWorker is not None and self.errorMapWorker.isRunning():
            self.errorMapWorker.cancel()
            self.errorMapWorker.wait()
        super().closeEvent(event)

    def setSignal(self, x, y):
        self.xAxisValuesOfOriginalSignal = np.asarray(x, dtype=float)
        self.yAxisValuesOfOriginalSignal = np.asarray(y, dtype=float)
        self.mainSignalDataPlotter.setData(self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal)
        if self.yAxisValuesOfOriginalSignal.size:
            self.ui.mainSignal.setLimits(yMin=self.yAxisValuesOfOriginalSignal.min(),
                                         yMax=self.yAxisValuesOfOriginalSignal.max())

    def read_file(self):
        path = QFileDialog.getOpenFileName()[0]
        if not path:
            return
        self.FileName = os.path.basename(path)
        if pathlib.Path(path).suffix != ".csv":
            self.statusBar().showMessage("only .csv files are supported")
            return
        data = np.genfromtxt(path, delimiter=',')
        self.setSignal(data[:ROW_LIMIT, 0], data[:ROW_LIMIT, 1])

    def chunks(self):
        x, y = self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal
        if x.size == 0:
            return
        self.chunkplotter.clear()
        self.chunkplotter.plot(x, y, pen="k")
        self.latexWriter.axes.clear()

        n_chunks = min(self.ui.SliderOfChunks.value(), x.size)
        if n_chunks > 0:
            fits = fitting.fit_chunks(x, y, n_chunks, self.ui.SliderOfOrder.value())
            rows = {'Chunk': [], 'equation': [], 'RMSE': []}
            for i, fit in enumerate(fits):
                x_plot = np.linspace(fit.x_min, fit.x_max, CHUNK_PLOT_POINTS)
                self.chunkplotter.plot(x_plot, fit.predict(x_plot), symbol='o')
                rows['Chunk'].append(f'eq of chunk {i + 1}')
                rows['equation'].append(f'${latex_equation(fit.coeffs)}$')
                rows['RMSE'].append(f'{fit.rmse:.4g}')
            table_frame = pd.DataFrame(rows)
            table = self.latexWriter.axes.table(cellText=table_frame.values, colLabels=table_frame.columns,
                                                loc='center')
            table.auto_set_column_width(col=list(range(len(table_frame.columns))))
            table.scale(1, 1.5)
            table.auto_set_font_size(True)

        self.latexWriter.fig.tight_layout()
        self.latexWriter.axes.set_axis_off()
        self.latexWriter.draw()

    def showHideErrorMap(self, state):
        self.errorMapCanv.setVisible(state == QtCore.Qt.Checked)

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
        if x.size == 0:
            return
        order = self.ui.SliderOfOrder.value()
        fraction = max(self.ui.SliderOfChunks.value() / 10, EXTRAPOLATION_MIN_FRACTION)
        # same point count as fitting.extrapolate: at least order+1 points
        n_fit = min(x.size, max(order + 1, int(round(fraction * x.size))))
        _, y_pred = fitting.extrapolate(x, y, fraction, order)
        x_line = x[n_fit - 1]
        self.verticalLinePlotter.setData([x_line, x_line], [y.min(), y.max()])
        self.extraPolationPlotter.setData(x, y_pred)

    def errorMapStart(self):
        if self.errorMapWorker is not None and self.errorMapWorker.isRunning():
            return
        x, y = self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal
        if x.size == 0:
            self.statusBar().showMessage("load data first")
            return
        max_order = self.ui.orderOfFittingPolynomialErrorMapSlider.value()
        max_chunks = self.ui.numberOfChunksErrorMapSlider.value()
        if max_order < 2 or max_chunks < 2:
            self.statusBar().showMessage("set both error map sliders to 2 or more")
            return
        fixed_order = self.ui.SliderOfOrder.value() or 5
        self.errorMapWorker = ErrorMapWorker(x.copy(), y.copy(), max_order, max_chunks, fixed_order, self)
        self.errorMapWorker.resultReady.connect(self.onErrorMapReady)
        self.errorMapWorker.start()
        self.statusBar().showMessage("computing error map...")

    def errorMapCancel(self):
        if self.errorMapWorker is not None and self.errorMapWorker.isRunning():
            self.errorMapWorker.cancel()
            self.statusBar().showMessage("cancelling after current map...")

    def onErrorMapReady(self, result):
        """GUI thread: normalise the raw maps and draw the selected pair."""
        if result is None:
            self.statusBar().showMessage("error map cancelled")
            return
        self.errorMapAxes = {0: result['chunks'], 1: result['orders'], 2: result['overlaps']}
        self.errorMaps = {key: normalize_to_percent(result[key]) for key in MAP_AXES}
        self.errorMapGenerated = True
        self.statusBar().showMessage("error map ready")
        self.showErrorMap()

    def _errorMapFor(self, x_axis, y_axis):
        """Normalised map with rows = y_axis, columns = x_axis."""
        for key, (row_axis, col_axis) in MAP_AXES.items():
            if {row_axis, col_axis} == {x_axis, y_axis}:
                m = self.errorMaps[key]
                return m if row_axis == y_axis else m.T
        raise ValueError("x and y axes must differ")

    def showErrorMap(self):
        x_axis = self.ui.xAxesComboBox.currentIndex()
        y_axis = self.ui.yAxesComboBox.currentIndex()
        if not self.errorMapGenerated or x_axis == y_axis:
            return
        ax = self.errorMapCanv.axes
        ax.clear()
        self.cax.clear()
        x_vals, y_vals = self.errorMapAxes[x_axis], self.errorMapAxes[y_axis]
        image = ax.imshow(self._errorMapFor(x_axis, y_axis), cmap="inferno", origin='lower',
                          extent=[*cell_extent(x_vals), *cell_extent(y_vals)])
        ax.set_xlabel(AXIS_NAMES[x_axis])
        ax.set_ylabel(AXIS_NAMES[y_axis])
        self.errorMapCanv.fig.colorbar(image, cax=self.cax, orientation='vertical')
        self.errorMapCanv.fig.tight_layout()
        self.errorMapCanv.draw()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = mainApp()
    main.show()
    sys.exit(app.exec_())
