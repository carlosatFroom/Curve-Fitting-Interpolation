import os
import pathlib
import sys
import threading

import matplotlib
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QSlider
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sympy import S, symbols, printing

import fitting
from mainapp import Ui_MainWindow

matplotlib.use('QT5Agg')

OVERLAPS = list(range(0, 26, 5))


def _error_grid(x, y, orders, chunk_counts, overlaps):
    """2-D RMSE grid; loops directly when fewer than two parameters vary (error_map needs two)."""
    seqs = [list(orders), list(chunk_counts), list(overlaps)]
    if sum(len(s) > 1 for s in seqs) == 2:
        return fitting.error_map(x, y, *seqs)
    grid = np.empty([len(s) for s in seqs])
    for idx in np.ndindex(grid.shape):
        o, c, v = (seqs[k][idx[k]] for k in range(3))
        fits = fitting.fit_chunks(x, y, int(c), int(o), float(v))
        grid[idx] = fitting.rmse(y, fitting.evaluate_piecewise(fits, x, blend=False))
    return grid.reshape(len(seqs[0]), len(seqs[1]) * len(seqs[2]))


def _normalized_pair(grid):
    """Grid as percent of its max, plus its transpose built from the same normalized data."""
    peak = grid.max()
    norm = grid / peak * 100 if peak > 0 else np.zeros_like(grid)
    return np.flip(norm, 0), np.flip(norm.T, 0)


class ErrorMapWorker(QtCore.QObject):
    """Computes numbers only; drawing happens on the GUI thread via `done`."""
    done = QtCore.pyqtSignal(object)  # (maxes, pairs) or None when cancelled
    failed = QtCore.pyqtSignal(str)

    def __init__(self, x, y, max_order, max_chunks, order, chunks, cancel):
        super().__init__()
        self.x, self.y = x, y
        self.max_order, self.max_chunks = max_order, max_chunks
        self.order, self.chunks = order, chunks
        self.cancel = cancel

    @QtCore.pyqtSlot()
    def run(self):
        orders = list(range(1, self.max_order + 1))
        chunk_counts = list(range(1, self.max_chunks + 1))
        jobs = [
            (orders, chunk_counts, [0]),                 # order x chunks
            ([self.order], chunk_counts, OVERLAPS),      # overlap x chunks
            (orders, [self.chunks], OVERLAPS),           # order x overlap
        ]
        try:
            grids = []
            for k, (orders_, chunks_, overlaps_) in enumerate(jobs):
                if self.cancel.is_set():
                    self.done.emit(None)
                    return
                grid = _error_grid(self.x, self.y, orders_, chunks_, overlaps_)
                if k == 1:  # rows = overlap, cols = chunks (as displayed)
                    grid = grid.reshape(len(chunks_), len(overlaps_)).T
                grids.append(_normalized_pair(grid))
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.failed.emit(str(exc))
            return
        self.done.emit(((self.max_order, self.max_chunks), grids))


class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, dpi=120):
        self.fig, self.axes = plt.subplots()
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.axes.set_facecolor((242 / 255, 243 / 255, 245 / 255))
        self.fig.set_facecolor((242 / 255, 243 / 255, 245 / 255))


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

        self.ui.showErrorMapCheckBox.stateChanged.connect(self.showHideErrorMap)
        self.ui.extrapolationCheckBox.stateChanged.connect(self.extraPolationToggle)
        self.ui.generateErrorMapBtn.clicked.connect(self.errorMapStart)
        self.ui.cancelBtn.clicked.connect(self.errorMapCancel)
        self.ui.xAxesComboBox.currentIndexChanged['int'].connect(self.showErrorMap)
        self.ui.yAxesComboBox.currentIndexChanged['int'].connect(self.showErrorMap)
        self.errorMapGenerated = False
        self.errorMapThread = None
        self.cancelEvent = threading.Event()
        divider = make_axes_locatable(self.errorMapCanv.axes)
        self.cax = divider.append_axes('right', size='5%', pad=0.05)
        self.cax.set_facecolor((242 / 255, 243 / 255, 245 / 255))

    def closeEvent(self, event):
        self.cancelEvent.set()
        if self.errorMapThread is not None:
            self.errorMapThread.wait()
        super().closeEvent(event)

    def errorMapStart(self):
        x, y = self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal
        if len(x) == 0 or (self.errorMapThread is not None and self.errorMapThread.isRunning()):
            return
        max_chunks = min(max(1, self.ui.numberOfChunksErrorMapSlider.value()), len(x))
        self.cancelEvent = threading.Event()
        self.errorMapThread = QtCore.QThread(self)
        self.errorMapWorker = ErrorMapWorker(
            x.copy(), y.copy(),
            max_order=max(1, self.ui.orderOfFittingPolynomialErrorMapSlider.value()),
            max_chunks=max_chunks,
            order=5,
            chunks=min(5, len(x)),
            cancel=self.cancelEvent,
        )
        self.errorMapWorker.moveToThread(self.errorMapThread)
        self.errorMapThread.started.connect(self.errorMapWorker.run)
        self.errorMapWorker.done.connect(self.onErrorMapReady)
        self.errorMapWorker.failed.connect(self.onErrorMapFailed)
        self.errorMapWorker.done.connect(self.errorMapThread.quit)
        self.errorMapWorker.failed.connect(self.errorMapThread.quit)
        self.errorMapThread.start()

    def errorMapCancel(self):
        self.cancelEvent.set()

    def onErrorMapReady(self, result):
        if result is None:
            return
        (self.maxPolyNomialOrder, self.maxNumberOfChunks), grids = result
        (self.polynomialOrderToNumberOfChunksError, self.numberOfChunksToPolynomialOrderError), \
            (self.overLappingPercentageToNumberOfChunksError, self.numberOfChunksToOverLappingPercentageError), \
            (self.polynomialOrderToOverlappingPercentageError, self.overLappingPercentageToPolynomialOrderError) = grids
        self.errorMapGenerated = True
        self.showErrorMap()

    def onErrorMapFailed(self, message):
        QMessageBox.warning(self, "Error map", message)

    def showHideErrorMap(self, state):
        self.errorMapCanv.setVisible(state == QtCore.Qt.Checked)

    def extraPolationToggle(self, state):

        if (state == QtCore.Qt.Checked):
            self.ui.SliderOfOrder.valueChanged.disconnect(self.chunks)
            self.ui.SliderOfChunks.valueChanged.disconnect(self.chunks)
            self.chunkplotter.clear()
            self.ui.chunksLabel.setText("extrapolation")
            self.ui.SliderOfChunks.valueChanged.connect(self.updateExtraPolation)
            self.ui.SliderOfOrder.valueChanged.connect(self.updateExtraPolation)
            self.mainSignalDataPlotter = self.ui.mainSignal.plot(self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal, pen="k")
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
        order = self.ui.SliderOfOrder.value()
        if len(x) < order + 1:
            return
        fraction = max(self.ui.SliderOfChunks.value(), 1) / 10
        _, y_pred = fitting.extrapolate(x, y, fraction, order)
        n_fit = min(len(x), max(order + 1, int(round(fraction * len(x)))))
        vertical_x = x[n_fit - 1]
        self.verticalLinePlotter.setData([vertical_x, vertical_x], [min(y), max(y)])
        self.extraPolationPlotter.setData(x, y_pred)

    def showErrorMap(self):
        if not self.errorMapGenerated:
            return

        self.errorMapCanv.axes.clear()
        self.cax.clear()

        xAxesComboBoxIndex = self.ui.xAxesComboBox.currentIndex()
        yAxesComboBoxIndex = self.ui.yAxesComboBox.currentIndex()

        maps = {
            (0, 1): (self.polynomialOrderToNumberOfChunksError, [0, self.maxNumberOfChunks, 0, self.maxPolyNomialOrder], "number of chunks", "polynomial order"),
            (1, 0): (self.numberOfChunksToPolynomialOrderError, [0, self.maxPolyNomialOrder, 0, self.maxNumberOfChunks], "polynomial order", "number of chunks"),
            (0, 2): (self.overLappingPercentageToNumberOfChunksError, [0, self.maxNumberOfChunks, 0, 25], "number of chunks", "overlapping percentage"),
            (2, 0): (self.numberOfChunksToOverLappingPercentageError, [0, 25, 0, self.maxNumberOfChunks], "overlapping percentage", "number of chunks"),
            (1, 2): (self.overLappingPercentageToPolynomialOrderError, [0, self.maxPolyNomialOrder, 0, 25], "polynomial order", "overlapping percentage"),
        }
        data, extent, xlabel, ylabel = maps.get(
            (xAxesComboBoxIndex, yAxesComboBoxIndex),
            (self.polynomialOrderToOverlappingPercentageError, [0, 25, 0, self.maxPolyNomialOrder], "overlapping percentage", "polynomial order"),
        )
        image = self.errorMapCanv.axes.imshow(data, cmap="inferno", extent=extent)
        self.errorMapCanv.axes.set_xlabel(xlabel)
        self.errorMapCanv.axes.set_ylabel(ylabel)

        self.errorMapCanv.fig.colorbar(image, cax=self.cax, orientation='vertical')
        self.errorMapCanv.fig.tight_layout()
        self.errorMapCanv.draw()

    def read_file(self):

        path = QFileDialog.getOpenFileName()[0]
        if not path:
            return
        self.FileName = os.path.basename(path)

        if pathlib.Path(path).suffix == ".csv":
            data = np.genfromtxt(path, delimiter=',')
            self.xAxisValuesOfOriginalSignal = data[:, 0][:1000]
            self.yAxisValuesOfOriginalSignal = data[:, 1][:1000]
            self.mainSignalDataPlotter.setData(self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal)
            self.ui.mainSignal.setLimits(yMin=min(self.yAxisValuesOfOriginalSignal), yMax=max(self.yAxisValuesOfOriginalSignal))

    def chunks(self):
        x, y = self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal
        if len(x) == 0:
            return
        self.chunkplotter.clear()
        self.chunkplotter.plot(x, y, pen="k")
        order = self.ui.SliderOfOrder.value()
        n_chunks = min(self.ui.SliderOfChunks.value(), len(x))
        if n_chunks < 1:
            return

        tableData = {'Chunk': [], 'equation': [], 'RMSE': []}
        x_sym = symbols("x")
        for chunkidx, fit in enumerate(fitting.fit_chunks(x, y, n_chunks, order)):
            xs = np.linspace(fit.x_min, fit.x_max, 50)
            self.chunkplotter.plot(xs, fit.predict(xs), symbol='o')
            poly = sum(S("{:6.2f}".format(v)) * x_sym ** i for i, v in enumerate(fit.coeffs[::-1]))
            tableData['RMSE'].append(round(fit.rmse, 3))
            tableData['equation'].append(f'${printing.latex(poly)}$')
            tableData['Chunk'].append(f'eq of chunk {chunkidx + 1}')

        self.latexWriter.axes.clear()
        if order > 0:
            TableData_frame = pd.DataFrame(tableData)
            table = self.latexWriter.axes.table(cellText=TableData_frame.values, colLabels=TableData_frame.columns, loc='center')
            table.auto_set_column_width(col=list(range(len(TableData_frame.columns))))
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
