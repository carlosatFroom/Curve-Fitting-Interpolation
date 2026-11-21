import os
import pathlib
import sys
import threading

import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QSlider
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from sympy import S, printing, symbols

import fitting
from mainapp import Ui_MainWindow

matplotlib.use('QT5Agg')

OVERLAP_PCTS = list(range(0, 26, 5))
MAX_OVERLAP_PCT = OVERLAP_PCTS[-1]
ERROR_MAP_ORDER = 5
ERROR_MAP_CHUNKS = 5
BG_COLOR = (242 / 255, 243 / 255, 245 / 255)


def normalize_to_percent(raw):
    peak = raw.max()
    if peak <= 0:
        return np.zeros_like(raw)
    return raw / peak * 100


def equation_latex(coeffs):
    x = symbols("x")
    poly = sum(S("{:6.2f}".format(c)) * x ** p for p, c in enumerate(coeffs[::-1]))
    return printing.latex(poly)


class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, dpi=120):
        self.fig, self.axes = plt.subplots()
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.axes.set_facecolor(BG_COLOR)
        self.fig.set_facecolor(BG_COLOR)


class mainApp(QtWidgets.QMainWindow):
    # Emitted from the error-map worker thread; the connected slot runs on the GUI thread.
    errorMapComputed = QtCore.pyqtSignal(str, object, object)

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

        self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal = [[], []]

        self.ui.showErrorMapCheckBox.stateChanged.connect(self.showHideErrorMap)
        self.ui.extrapolationCheckBox.stateChanged.connect(self.extraPolationToggle)
        self.ui.generateErrorMapBtn.clicked.connect(self.errorMapStart)
        self.ui.cancelBtn.clicked.connect(self.errorMapCancel)
        self.ui.xAxesComboBox.currentIndexChanged['int'].connect(self.showErrorMap)
        self.ui.yAxesComboBox.currentIndexChanged['int'].connect(self.showErrorMap)

        self.errorMapComputed.connect(self.onErrorMapComputed)
        self.errorMaps = {}
        self.errorMapGenerated = False
        self.maxPolyNomialOrder = 0
        self.maxNumberOfChunks = 0
        self._errorMapCancel = threading.Event()

        divider = make_axes_locatable(self.errorMapCanv.axes)
        self.cax = divider.append_axes('right', size='5%', pad=0.05)
        self.cax.set_facecolor(BG_COLOR)

    def errorMapStart(self):
        self.maxPolyNomialOrder = self.ui.orderOfFittingPolynomialErrorMapSlider.value()
        self.maxNumberOfChunks = self.ui.numberOfChunksErrorMapSlider.value()
        self.errorMaps = {}
        self.errorMapGenerated = False
        self._errorMapCancel = threading.Event()
        x = np.asarray(self.xAxisValuesOfOriginalSignal, dtype=float)
        y = np.asarray(self.yAxisValuesOfOriginalSignal, dtype=float)
        self.errorMapProcessor = threading.Thread(
            target=self._errorMapWorker,
            args=(x, y, self.maxPolyNomialOrder, self.maxNumberOfChunks, self._errorMapCancel),
            daemon=True)
        self.errorMapProcessor.start()

    def errorMapCancel(self):
        self._errorMapCancel.set()

    def _errorMapWorker(self, x, y, maxOrder, maxChunks, cancel):
        # Worker thread: numpy only. No Qt or matplotlib calls here.
        orders = list(range(1, maxOrder + 1))
        chunkCounts = list(range(1, maxChunks + 1))
        jobs = [
            ("orderChunks", orders, chunkCounts, [0]),
            ("chunksOverlap", [ERROR_MAP_ORDER], chunkCounts, OVERLAP_PCTS),
            ("orderOverlap", orders, [ERROR_MAP_CHUNKS], OVERLAP_PCTS),
        ]
        for name, orderVals, chunkVals, overlapVals in jobs:
            if cancel.is_set():
                return
            # fitting.error_map needs exactly two varying sequences
            if sum(len(v) > 1 for v in (orderVals, chunkVals, overlapVals)) != 2:
                continue
            raw = fitting.error_map(x, y, orderVals, chunkVals, overlapVals)
            if cancel.is_set():
                return
            norm = normalize_to_percent(raw)
            # raw axis 0 is the first varying arg in (order, chunks, overlap)
            matrix = norm.T if name == "chunksOverlap" else norm  # rows = vertical axis
            self.errorMapComputed.emit(name, np.flip(matrix, 0), np.flip(matrix.T, 0))

    def onErrorMapComputed(self, name, display, transposed):
        self.errorMaps[name] = (display, transposed)
        self.errorMapGenerated = True
        self.showErrorMap()

    def showHideErrorMap(self, state):
        if (state == QtCore.Qt.Checked):
            self.errorMapCanv.setVisible(True)
        else:
            self.errorMapCanv.setVisible(False)

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
        x = np.asarray(self.xAxisValuesOfOriginalSignal, dtype=float)
        y = np.asarray(self.yAxisValuesOfOriginalSignal, dtype=float)
        numberOfChunks = self.ui.SliderOfChunks.value()
        polynomialOrder = self.ui.SliderOfOrder.value()
        if numberOfChunks == 0 or x.size == 0:
            self.verticalLinePlotter.setData([], [])
            self.extraPolationPlotter.setData([], [])
            return

        fitFraction = numberOfChunks / 10
        fitEndIndex = min(x.size, max(polynomialOrder + 1, int(fitFraction * x.size)))
        _, yPred = fitting.extrapolate(x, y, fitFraction, polynomialOrder)
        verticalLineX = x[fitEndIndex - 1]
        self.verticalLinePlotter.setData([verticalLineX, verticalLineX], [y.min(), y.max()])
        self.extraPolationPlotter.setData(x, yPred)

    def showErrorMap(self):
        if not self.errorMapGenerated:
            return

        maxO, maxC, maxOv = self.maxPolyNomialOrder, self.maxNumberOfChunks, MAX_OVERLAP_PCT
        xAxesComboBoxIndex = self.ui.xAxesComboBox.currentIndex()
        yAxesComboBoxIndex = self.ui.yAxesComboBox.currentIndex()

        if (xAxesComboBoxIndex, yAxesComboBoxIndex) == (0, 1):
            key, part, extent = "orderChunks", 0, [0, maxC, 0, maxO]
            xLabel, yLabel = "number of chunks", "polynomial order"
        elif (xAxesComboBoxIndex, yAxesComboBoxIndex) == (1, 0):
            key, part, extent = "orderChunks", 1, [0, maxO, 0, maxC]
            xLabel, yLabel = "polynomial order", "number of chunks"
        elif (xAxesComboBoxIndex, yAxesComboBoxIndex) == (0, 2):
            key, part, extent = "chunksOverlap", 0, [0, maxC, 0, maxOv]
            xLabel, yLabel = "number of chunks", "overlapping percentage"
        elif (xAxesComboBoxIndex, yAxesComboBoxIndex) == (2, 0):
            key, part, extent = "chunksOverlap", 1, [0, maxOv, 0, maxC]
            xLabel, yLabel = "overlapping percentage", "number of chunks"
        elif (xAxesComboBoxIndex, yAxesComboBoxIndex) == (1, 2):
            key, part, extent = "orderOverlap", 1, [0, maxO, 0, maxOv]
            xLabel, yLabel = "polynomial order", "overlapping percentage"
        else:
            key, part, extent = "orderOverlap", 0, [0, maxOv, 0, maxO]
            xLabel, yLabel = "overlapping percentage", "polynomial order"

        if key not in self.errorMaps:
            return  # map was skipped (e.g. only one order or one chunk count)

        self.errorMapCanv.axes.clear()
        self.cax.clear()
        image = self.errorMapCanv.axes.imshow(self.errorMaps[key][part], cmap="inferno", extent=extent)
        self.errorMapCanv.axes.set_xlabel(xLabel)
        self.errorMapCanv.axes.set_ylabel(yLabel)
        self.errorMapCanv.fig.colorbar(image, cax=self.cax, orientation='vertical')
        self.errorMapCanv.fig.tight_layout()
        self.errorMapCanv.draw()

    def read_file(self):

        path = QFileDialog.getOpenFileName()[0]
        if not path:
            return
        self.FileName = os.path.basename(path)

        if pathlib.Path(path).suffix == ".csv":
            self.data = np.genfromtxt(path, delimiter=',')
            self.xAxisValuesOfOriginalSignal = self.data[:, 0][:1000]
            self.yAxisValuesOfOriginalSignal = self.data[:, 1][:1000]
            self.mainSignalDataPlotter.setData(self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal)

        self.ui.mainSignal.setLimits(yMin=min(self.yAxisValuesOfOriginalSignal), yMax=max(self.yAxisValuesOfOriginalSignal))

    def chunks(self):
        numberOfChunks = self.ui.SliderOfChunks.value()
        order = self.ui.SliderOfOrder.value()
        x = np.asarray(self.xAxisValuesOfOriginalSignal, dtype=float)
        y = np.asarray(self.yAxisValuesOfOriginalSignal, dtype=float)

        self.chunkplotter.clear()
        self.chunkplotter.plot(x, y, pen="k")
        self.latexWriter.axes.clear()

        fits = []
        if x.size and numberOfChunks > 0:
            try:
                fits = fitting.fit_chunks(x, y, numberOfChunks, order)
            except ValueError:
                fits = []  # more chunks than points

        yRange = float(y.max() - y.min()) if x.size else 0.0
        tableData = {'Chunk': [], 'equation': [], 'rmse': [], 'error%': []}
        for chunkidx, fit in enumerate(fits):
            xs = np.linspace(fit.x_min, fit.x_max)
            self.chunkplotter.plot(xs, fit.predict(xs), symbol='o')
            # error% = chunk RMSE as % of the whole signal's y-range
            errorPct = fit.rmse / yRange * 100 if yRange > 0 else 0.0
            tableData['Chunk'].append(f'eq of chunk {chunkidx + 1}')
            tableData['equation'].append(f'${equation_latex(fit.coeffs)}$')
            tableData['rmse'].append(round(fit.rmse, 4))
            tableData['error%'].append(round(errorPct, 3))

        if fits and order > 0:  # plot the table only if the spin box has value
            tableFrame = pd.DataFrame(tableData)
            table = self.latexWriter.axes.table(cellText=tableFrame.values, colLabels=tableFrame.columns, loc='center')
            table.auto_set_column_width(col=list(range(len(tableFrame.columns))))
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
