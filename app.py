import os
import pathlib
import sys
import threading

import matplotlib
matplotlib.use('QT5Agg')
import numpy as np
import pandas as pd
import pyqtgraph as pg  # noqa: F401
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QSlider
from sympy import S, symbols, printing

import fitting
from mainapp import Ui_MainWindow

OVERLAP_PCTS = list(range(0, 26, 5))
DEFAULT_ORDER = 5
DEFAULT_CHUNKS = 5


class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, dpi=120):
        self.fig, self.axes = plt.subplots()
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.axes.set_facecolor((242/255, 243/255, 245/255))
        self.fig.set_facecolor((242/255, 243/255, 245/255))


class mainApp(QtWidgets.QMainWindow):
    errorMapReady = QtCore.pyqtSignal(object)

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

        self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal = np.array([]), np.array([])

        self.ui.showErrorMapCheckBox.stateChanged.connect(self.showHideErrorMap)
        self.ui.extrapolationCheckBox.stateChanged.connect(self.extraPolationToggle)
        self.ui.generateErrorMapBtn.clicked.connect(self.errorMapStart)
        self.ui.cancelBtn.clicked.connect(self.errorMapCancel)
        self.ui.xAxesComboBox.currentIndexChanged['int'].connect(self.showErrorMap)
        self.ui.yAxesComboBox.currentIndexChanged['int'].connect(self.showErrorMap)
        self.errorMapReady.connect(self.onErrorMapReady)
        self.errorMapGenerated = False
        self.stopFlag = False
        divider = make_axes_locatable(self.errorMapCanv.axes)
        self.cax = divider.append_axes('right', size='5%', pad=0.05)
        self.cax.set_facecolor((242/255, 243/255, 245/255))

    def hasData(self):
        return len(self.xAxisValuesOfOriginalSignal) > 0

    def errorMapStart(self):
        if not self.hasData():
            return
        self.stopFlag = False
        maxOrder = self.ui.orderOfFittingPolynomialErrorMapSlider.value()
        maxChunks = self.ui.numberOfChunksErrorMapSlider.value()
        x, y = self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal
        self.errorMapProcessor = threading.Thread(
            target=self.generateErrorMap, args=(x, y, maxOrder, maxChunks), daemon=True)
        self.errorMapProcessor.start()

    def errorMapCancel(self):
        self.stopFlag = True

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
        if not self.hasData():
            return
        x, y = self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal
        fraction = max(self.ui.SliderOfChunks.value(), 1) / 10
        order = self.ui.SliderOfOrder.value()
        _, yPred = fitting.extrapolate(x, y, fraction, order)
        nFit = max(int(len(x) * fraction), order + 1)
        xLine = x[min(nFit, len(x)) - 1]
        self.verticalLinePlotter.setData([xLine, xLine], [np.min(y), np.max(y)])
        self.extraPolationPlotter.setData(x, yPred)

    def showErrorMap(self):
        if not self.errorMapGenerated:
            return

        self.errorMapCanv.axes.clear()
        self.cax.clear()

        xi = self.ui.xAxesComboBox.currentIndex()
        yi = self.ui.yAxesComboBox.currentIndex()
        maxO, maxC = self.maxPolyNomialOrder, self.maxNumberOfChunks
        axes = self.errorMapCanv.axes
        cmap = "inferno"

        if xi == 0 and yi == 1:
            image = axes.imshow(self.polynomialOrderToNumberOfChunksError, cmap=cmap, extent=[0, maxC, 0, maxO])
            axes.set_xlabel("number of chunks")
            axes.set_ylabel("polynomial order")
        elif xi == 1 and yi == 0:
            image = axes.imshow(self.numberOfChunksToPolynomialOrderError, cmap=cmap, extent=[0, maxO, 0, maxC])
            axes.set_xlabel("polynomial order")
            axes.set_ylabel("number of chunks")
        elif xi == 0 and yi == 2:
            image = axes.imshow(self.overLappingPercentageToNumberOfChunksError, cmap=cmap, extent=[0, maxC, 0, 25])
            axes.set_xlabel("number of chunks")
            axes.set_ylabel("overlapping percentage")
        elif xi == 2 and yi == 0:
            image = axes.imshow(self.numberOfChunksToOverLappingPercentageError, cmap=cmap, extent=[0, 25, 0, maxC])
            axes.set_xlabel("overlapping percentage")
            axes.set_ylabel("number of chunks")
        elif xi == 1 and yi == 2:
            image = axes.imshow(self.overLappingPercentageToPolynomialOrderError, cmap=cmap, extent=[0, maxO, 0, 25])
            axes.set_xlabel("polynomial order")
            axes.set_ylabel("overlapping percentage")
        else:
            image = axes.imshow(self.polynomialOrderToOverlappingPercentageError, cmap=cmap, extent=[0, 25, 0, maxO])
            axes.set_xlabel("overlapping percentage")
            axes.set_ylabel("polynomial order")

        self.errorMapCanv.fig.colorbar(image, cax=self.cax, orientation='vertical')
        self.errorMapCanv.fig.tight_layout()
        self.errorMapCanv.draw()

    @staticmethod
    def normalizeAndPrepData(data):
        peak = data.max()
        normalized = data / peak * 100 if peak > 0 else data.copy()
        return [np.flip(normalized, 0), np.flip(np.transpose(normalized), 0)]

    def generateErrorMap(self, x, y, maxOrder, maxChunks):
        """Runs on a worker thread: numeric work only, results go to the GUI thread via signal."""
        try:
            maxChunks = min(maxChunks, len(x))
            orders = list(range(1, maxOrder + 1))
            chunkCounts = list(range(1, maxChunks + 1))
            fixedChunks = min(DEFAULT_CHUNKS, len(x))
            results = {"maxOrder": maxOrder, "maxChunks": maxChunks}

            if self.stopFlag:
                return
            data = fitting.error_map(x, y, orders, chunkCounts, [0])
            results["orderChunks"] = self.normalizeAndPrepData(data)

            if self.stopFlag:
                return
            data = fitting.error_map(x, y, [DEFAULT_ORDER], chunkCounts, OVERLAP_PCTS).T
            results["overlapChunks"] = self.normalizeAndPrepData(data)

            if self.stopFlag:
                return
            data = fitting.error_map(x, y, orders, [fixedChunks], OVERLAP_PCTS)
            results["orderOverlap"] = self.normalizeAndPrepData(data)
        except Exception as exc:
            print("error map failed:", exc, file=sys.stderr)
            return
        if not self.stopFlag:
            self.errorMapReady.emit(results)

    def onErrorMapReady(self, results):
        self.maxPolyNomialOrder = results["maxOrder"]
        self.maxNumberOfChunks = results["maxChunks"]
        [self.polynomialOrderToNumberOfChunksError, self.numberOfChunksToPolynomialOrderError] = results["orderChunks"]
        [self.overLappingPercentageToNumberOfChunksError, self.numberOfChunksToOverLappingPercentageError] = results["overlapChunks"]
        [self.polynomialOrderToOverlappingPercentageError, self.overLappingPercentageToPolynomialOrderError] = results["orderOverlap"]
        self.errorMapGenerated = True
        self.showErrorMap()

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

        if self.hasData():
            self.ui.mainSignal.setLimits(yMin=np.min(self.yAxisValuesOfOriginalSignal),
                                         yMax=np.max(self.yAxisValuesOfOriginalSignal))

    def chunks(self):
        tableData = {'Chunk': [], 'equation': [], 'rmse': []}
        self.numberofChunkies = self.ui.SliderOfChunks.value()
        self.chunkplotter.clear()
        self.chunkplotter.plot(self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal, pen="k")
        order = self.ui.SliderOfOrder.value()

        if self.numberofChunkies > 0 and self.hasData():
            x, y = self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal
            nChunks = min(self.numberofChunkies, len(x))
            fits = fitting.fit_chunks(x, y, nChunks, order)
            symbol = symbols("x")

            for chunkidx, fit in enumerate(fits):
                xs = np.linspace(fit.x_min, fit.x_max)
                self.chunkplotter.plot(xs, fit.predict(xs), symbol='o')

                poly = sum(S("{:6.2f}".format(v)) * symbol ** i for i, v in enumerate(fit.coeffs[::-1]))
                tableData['rmse'].append(round(fit.rmse, 3))
                tableData['equation'].append(f'${printing.latex(poly)}$')
                tableData['Chunk'].append(f'eq of chunk {chunkidx + 1}')

            self.latexWriter.axes.clear()

            if order > 0:
                frame = pd.DataFrame(tableData)
                table = self.latexWriter.axes.table(cellText=frame.values, colLabels=frame.columns, loc='center')
                table.auto_set_column_width(col=list(range(len(frame.columns))))
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
