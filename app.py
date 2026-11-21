import os
import sys
import threading
from concurrent.futures import CancelledError

import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QSlider

import fitting
from mainapp import Ui_MainWindow

OVERLAP_VALUES = list(range(0, 26, 5))
DEFAULT_ORDER = 5
DEFAULT_CHUNKS = 5


def polynomial_latex(coeffs):
    terms = []
    for power, c in zip(range(len(coeffs) - 1, -1, -1), coeffs):
        if abs(c) < 0.005 and len(coeffs) > 1:
            continue
        mono = "" if power == 0 else ("x" if power == 1 else f"x^{{{power}}}")
        terms.append(f"{c:.2f}{mono}")
    return " + ".join(terms).replace("+ -", "- ") or "0"


class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self,parent=None, dpi = 120):
        self.fig, self.axes = plt.subplots()
        super(MatplotlibCanvas,self).__init__(self.fig)
        self.axes.set_facecolor((242/255,243/255,245/255))
        self.fig.set_facecolor((242/255,243/255,245/255))


class mainApp(QtWidgets.QMainWindow):

    errorMapReady = QtCore.pyqtSignal(object)
    AXIS_NAMES = ["number of chunks", "polynomial order", "overlapping percentage"]

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



        self.mainSignalDataPlotter = self.ui.mainSignal.plot([], [], pen = "k")
        self.verticalLinePlotter = self.ui.mainSignal.plot([],[], pen = "g")
        self.extraPolationPlotter = self.ui.mainSignal.plot([],[], pen = "r")


        self.chunkplotter = self.ui.mainSignal
        self.ui.actionopen.triggered.connect(lambda :self.read_file())
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

        self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal = [[],[]]

        self.ui.showErrorMapCheckBox.stateChanged.connect(self.showHideErrorMap)
        self.ui.extrapolationCheckBox.stateChanged.connect(self.extraPolationToggle)
        # self.ui.extrapolationCheckBox.stateChanged.connect()
        self.ui.generateErrorMapBtn.clicked.connect(self.errorMapStart)
        self.ui.cancelBtn.clicked.connect(self.errorMapCancel)
        self.ui.xAxesComboBox.currentIndexChanged['int'].connect(self.showErrorMap)
        self.ui.yAxesComboBox.currentIndexChanged['int'].connect(self.showErrorMap)
        self.errorMapGenerated = False
        self.errorMaps = {}
        self.errorMapProcessor = None
        self.stopFlag = False
        self.errorMapReady.connect(self.onErrorMapReady)
        divider = make_axes_locatable(self.errorMapCanv.axes)
        self.cax = divider.append_axes('right', size='5%', pad=0.05)
        self.cax.set_facecolor((242/255,243/255,245/255))
        

    def errorMapStart(self):
        if len(self.xAxisValuesOfOriginalSignal) < 2:
            return
        if self.errorMapProcessor is not None and self.errorMapProcessor.is_alive():
            return
        self.stopFlag = False
        self.errorMapProcessor = threading.Thread(target=self.generateErrorMap, daemon=True)
        self.errorMapProcessor.start()

    def errorMapCancel(self):
        self.stopFlag = True

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
            self.mainSignalDataPlotter = self.ui.mainSignal.plot(self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal, pen = "k")
            self.verticalLinePlotter = self.ui.mainSignal.plot([],[], pen = "g")
            self.extraPolationPlotter = self.ui.mainSignal.plot([],[], pen = "r")
            self.updateExtraPolation()

        else:
            self.ui.SliderOfOrder.valueChanged.disconnect(self.updateExtraPolation)
            self.ui.SliderOfChunks.valueChanged.disconnect(self.updateExtraPolation)
            self.ui.chunksLabel.setText("Chunks")
            self.ui.SliderOfChunks.valueChanged.connect(self.chunks)
            self.ui.SliderOfOrder.valueChanged.connect(self.chunks)
            self.verticalLinePlotter.setData([],[])
            self.extraPolationPlotter.setData([],[])
            self.chunks()

    def updateExtraPolation(self):
        x, y = self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal
        if len(x) == 0:
            return
        order = self.ui.SliderOfOrder.value()
        fraction = max(self.ui.SliderOfChunks.value() / 10, 1e-9)
        _, predicted = fitting.extrapolate(x, y, fraction, order)
        lastFitted = fitting._extrapolation_count(len(x), fraction, order) - 1
        self.verticalLinePlotter.setData([x[lastFitted], x[lastFitted]], [min(y), max(y)])
        self.extraPolationPlotter.setData(x, predicted)

    def showErrorMap(self):
        if not self.errorMapGenerated:
            return

        xIndex = self.ui.xAxesComboBox.currentIndex()
        yIndex = self.ui.yAxesComboBox.currentIndex()
        if xIndex == yIndex:
            return

        self.errorMapCanv.axes.clear()
        self.cax.clear()

        image, extent = self.errorMaps[(xIndex, yIndex)]
        shown = self.errorMapCanv.axes.imshow(image, cmap="inferno", extent=extent, aspect="auto")
        self.errorMapCanv.axes.set_xlabel(self.AXIS_NAMES[xIndex])
        self.errorMapCanv.axes.set_ylabel(self.AXIS_NAMES[yIndex])
        self.errorMapCanv.fig.colorbar(shown, cax=self.cax, orientation='vertical')
        self.errorMapCanv.fig.tight_layout()
        self.errorMapCanv.draw()

    def storeErrorMap(self, errors, axisA, axisB, valuesA, valuesB):
        """errors[a, b] -> normalised (0-100) images for both axis orientations."""
        normalized = errors / errors.max() * 100 if errors.max() > 0 else errors

        def extent(values, index):
            step = 5 if index == 2 else 1
            return [values[0] - step / 2, values[-1] + step / 2]

        # x = B, y = A (rows flipped so y grows upward)
        self.errorMaps[(axisB, axisA)] = (np.flipud(normalized), extent(valuesB, axisB) + extent(valuesA, axisA))
        self.errorMaps[(axisA, axisB)] = (np.flipud(normalized.T), extent(valuesA, axisA) + extent(valuesB, axisB))

    def generateErrorMap(self):
        """Runs on a worker thread: computes only, never touches Qt or matplotlib."""
        n = len(self.xAxisValuesOfOriginalSignal)
        maxOrder = max(2, self.ui.orderOfFittingPolynomialErrorMapSlider.value())
        maxChunks = min(max(2, self.ui.numberOfChunksErrorMapSlider.value()), n)
        if maxChunks < 2:
            return
        orders, chunks = list(range(1, maxOrder + 1)), list(range(1, maxChunks + 1))
        x, y, cancel = self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal, lambda: self.stopFlag
        try:
            maps = {
                (1, 0): (fitting.error_map(x, y, orders, chunks, [0], should_cancel=cancel), orders, chunks),
                (0, 2): (fitting.error_map(x, y, [DEFAULT_ORDER], chunks, OVERLAP_VALUES, should_cancel=cancel),
                         chunks, OVERLAP_VALUES),
                (1, 2): (fitting.error_map(x, y, orders, [min(DEFAULT_CHUNKS, n)], OVERLAP_VALUES, should_cancel=cancel),
                         orders, OVERLAP_VALUES),
            }
        except CancelledError:
            return
        self.errorMapReady.emit((maps, maxOrder, maxChunks))

    def onErrorMapReady(self, payload):
        maps, self.maxPolyNomialOrder, self.maxNumberOfChunks = payload
        self.errorMaps = {}
        for (axisA, axisB), (errors, valuesA, valuesB) in maps.items():
            self.storeErrorMap(errors, axisA, axisB, valuesA, valuesB)
        self.errorMapGenerated = True
        self.showErrorMap()

    def read_file(self):

        path = QFileDialog.getOpenFileName()[0]
        if not path or os.path.splitext(path)[1].lower() != ".csv":
            return
        self.FileName = os.path.basename(path)

        data = np.genfromtxt(path, delimiter=',', usecols=(0, 1), ndmin=2)
        data = data[np.isfinite(data).all(axis=1)][:1000]
        if len(data) == 0:
            return
        self.xAxisValuesOfOriginalSignal = data[:, 0]
        self.yAxisValuesOfOriginalSignal = data[:, 1]
        self.errorMapGenerated = False
        self.mainSignalDataPlotter.setData(self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal)

        self.ui.mainSignal.setLimits(yMin = min(self.yAxisValuesOfOriginalSignal), yMax = max(self.yAxisValuesOfOriginalSignal))

    def chunks(self):
        x, y = self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal
        numberOfChunks = self.ui.SliderOfChunks.value()
        order = self.ui.SliderOfOrder.value()
        self.chunkplotter.clear()
        self.chunkplotter.plot(x, y, pen = "k")
        if numberOfChunks < 1 or numberOfChunks > len(x):
            return

        tableData = []
        for chunkidx, fit in enumerate(fitting.fit_chunks(x, y, numberOfChunks, order)):
            xs = np.linspace(fit.x_min, fit.x_max)
            self.chunkplotter.plot(xs, fit.predict(xs), symbol='o')
            tableData.append([f'eq of chunk {chunkidx + 1}', f'${polynomial_latex(fit.coeffs)}$', round(fit.rmse, 3)])

        self.latexWriter.axes.clear()
        if order > 0:
            table = self.latexWriter.axes.table(cellText=tableData, colLabels=['Chunk', 'equation', 'RMSE'], loc='center')
            table.auto_set_column_width(col=[0, 1, 2])
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
