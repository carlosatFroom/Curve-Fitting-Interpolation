import os
import sys
import pathlib
import threading

import numpy as np
import pyqtgraph as pg
import matplotlib
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QSlider
from sympy import S, symbols, printing
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

from mainapp import Ui_MainWindow
import fitting

matplotlib.use('QT5Agg')


class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, dpi=120):
        self.fig, self.axes = plt.subplots()
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.axes.set_facecolor((242 / 255, 243 / 255, 245 / 255))
        self.fig.set_facecolor((242 / 255, 243 / 255, 245 / 255))


class mainApp(QtWidgets.QMainWindow):

    # Emitted from the worker thread when error-map data is ready; the slot runs
    # on the GUI thread so all matplotlib drawing stays on the main thread.
    errorMapReady = QtCore.pyqtSignal()

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

        self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal = [np.array([]), np.array([])]

        self.ui.showErrorMapCheckBox.stateChanged.connect(self.showHideErrorMap)
        self.ui.extrapolationCheckBox.stateChanged.connect(self.extraPolationToggle)
        self.ui.generateErrorMapBtn.clicked.connect(self.errorMapStart)
        self.ui.cancelBtn.clicked.connect(self.errorMapCancel)
        self.ui.xAxesComboBox.currentIndexChanged['int'].connect(self.showErrorMap)
        self.ui.yAxesComboBox.currentIndexChanged['int'].connect(self.showErrorMap)
        self.errorMapGenerated = False
        self.stopFlag = False
        self.errorMapReady.connect(self._onErrorMapReady)

        divider = make_axes_locatable(self.errorMapCanv.axes)
        self.cax = divider.append_axes('right', size='5%', pad=0.05)
        self.cax.set_facecolor((242 / 255, 243 / 255, 245 / 255))

    # ------------------------------------------------------------------ #
    # Error map (computed off-thread, drawn on the GUI thread)
    # ------------------------------------------------------------------ #
    def errorMapStart(self):
        self.stopFlag = False
        self.errorMapProcessor = threading.Thread(target=self.generateErrorMap)
        self.errorMapProcessor.daemon = True
        self.errorMapProcessor.start()

    def errorMapCancel(self):
        self.stopFlag = True

    def showHideErrorMap(self, state):
        self.errorMapCanv.setVisible(state == QtCore.Qt.Checked)

    def _hasData(self):
        return len(self.xAxisValuesOfOriginalSignal) > 1

    def generateErrorMap(self):
        """Runs on a worker thread: pure-numpy math only, no Qt/matplotlib."""
        if not self._hasData():
            return
        x = self.xAxisValuesOfOriginalSignal
        y = self.yAxisValuesOfOriginalSignal

        self.maxPolyNomialOrder = self.ui.orderOfFittingPolynomialErrorMapSlider.value()
        self.maxNumberOfChunks = self.ui.numberOfChunksErrorMapSlider.value()
        max_order = max(1, self.maxPolyNomialOrder)
        max_chunks = max(1, min(self.maxNumberOfChunks, len(x)))

        orders = list(range(1, max_order + 1))
        chunks = list(range(1, max_chunks + 1))
        overlaps = list(range(0, 26, 5))

        if self.stopFlag:
            return
        # order x chunks  (axis0=order, axis1=chunks)
        order_chunks = fitting.error_map(x, y, orders, chunks, [0.0])
        if self.stopFlag:
            return
        # chunks x overlap (axis0=chunks, axis1=overlap) -> transpose to overlap x chunks
        chunks_overlap = fitting.error_map(x, y, [5], chunks, overlaps)
        overlap_chunks = chunks_overlap.T
        if self.stopFlag:
            return
        # order x overlap (axis0=order, axis1=overlap)
        order_overlap = fitting.error_map(x, y, orders, [5], overlaps)

        (self.polynomialOrderToNumberOfChunksError,
         self.numberOfChunksToPolynomialOrderError) = self.normalizeAndPrepData(order_chunks)
        (self.overLappingPercentageToNumberOfChunksError,
         self.numberOfChunksToOverLappingPercentageError) = self.normalizeAndPrepData(overlap_chunks)
        (self.overLappingPercentageToPolynomialOrderError,
         self.polynomialOrderToOverlappingPercentageError) = self.normalizeAndPrepData(order_overlap.T)

        self.errorMapGenerated = True
        self.errorMapReady.emit()

    def _onErrorMapReady(self):
        # Runs on the GUI thread.
        self.showErrorMap()

    def normalizeAndPrepData(self, data):
        normalized = self.normalizeData(np.array(data, dtype=float))
        normalized_flipped = np.flip(normalized, 0)
        transposed_flipped = np.flip(np.transpose(normalized), 0)
        return [normalized_flipped, transposed_flipped]

    def normalizeData(self, data):
        data = np.array(data, dtype=float)
        m = data.max()
        if m != 0:
            data = data / m * 100
        return data

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
        if not self._hasData():
            return
        x = self.xAxisValuesOfOriginalSignal
        y = self.yAxisValuesOfOriginalSignal
        n = len(x)
        order = self.ui.SliderOfOrder.value()

        # fraction of the signal used to fit (slider 0..10 -> up to all points)
        frac = np.clip(self.ui.SliderOfChunks.value() / 10.0, 1e-6, 1.0)
        fit_count = int(frac * n)
        fit_count = max(fit_count, order + 1)
        fit_count = min(fit_count, n)
        verticalLineX = x[fit_count - 1]

        _, y_pred = fitting.extrapolate(x, y, fit_count / n, order)
        self.verticalLinePlotter.setData(
            [verticalLineX, verticalLineX],
            [float(np.min(y)), float(np.max(y))])
        self.extraPolationPlotter.setData(x, y_pred)

    def showErrorMap(self):
        if not self.errorMapGenerated:
            return

        self.errorMapCanv.axes.clear()
        self.cax.clear()

        xi = self.ui.xAxesComboBox.currentIndex()
        yi = self.ui.yAxesComboBox.currentIndex()

        if xi == 0 and yi == 1:
            image = self.errorMapCanv.axes.imshow(
                self.polynomialOrderToNumberOfChunksError, cmap="inferno",
                extent=[0, self.maxNumberOfChunks, 0, self.maxPolyNomialOrder])
            self.errorMapCanv.axes.set_xlabel("number of chunks")
            self.errorMapCanv.axes.set_ylabel("polynomial order")
        elif xi == 1 and yi == 0:
            image = self.errorMapCanv.axes.imshow(
                self.numberOfChunksToPolynomialOrderError, cmap="inferno",
                extent=[0, self.maxPolyNomialOrder, 0, self.maxNumberOfChunks])
            self.errorMapCanv.axes.set_xlabel("polynomial order")
            self.errorMapCanv.axes.set_ylabel("number of chunks")
        elif xi == 0 and yi == 2:
            image = self.errorMapCanv.axes.imshow(
                self.overLappingPercentageToNumberOfChunksError, cmap="inferno",
                extent=[0, self.maxNumberOfChunks, 0, 25])
            self.errorMapCanv.axes.set_xlabel("number of chunks")
            self.errorMapCanv.axes.set_ylabel("overlapping percentage")
        elif xi == 2 and yi == 0:
            image = self.errorMapCanv.axes.imshow(
                self.numberOfChunksToOverLappingPercentageError, cmap="inferno",
                extent=[0, 25, 0, self.maxNumberOfChunks])
            self.errorMapCanv.axes.set_xlabel("overlapping percentage")
            self.errorMapCanv.axes.set_ylabel("number of chunks")
        elif xi == 1 and yi == 2:
            image = self.errorMapCanv.axes.imshow(
                self.overLappingPercentageToPolynomialOrderError, cmap="inferno",
                extent=[0, self.maxPolyNomialOrder, 0, 25])
            self.errorMapCanv.axes.set_xlabel("polynomial order")
            self.errorMapCanv.axes.set_ylabel("overlapping percentage")
        else:
            image = self.errorMapCanv.axes.imshow(
                self.polynomialOrderToOverlappingPercentageError, cmap="inferno",
                extent=[0, 25, 0, self.maxPolyNomialOrder])
            self.errorMapCanv.axes.set_xlabel("overlapping percentage")
            self.errorMapCanv.axes.set_ylabel("polynomial order")

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
            self.mainSignalDataPlotter.setData(
                self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal)

        self.ui.mainSignal.setLimits(
            yMin=float(np.min(self.yAxisValuesOfOriginalSignal)),
            yMax=float(np.max(self.yAxisValuesOfOriginalSignal)))

    def chunks(self):
        if not self._hasData():
            return
        tableData = {'Chunk': [], 'equation': [], 'error': []}
        self.numberofChunkies = self.ui.SliderOfChunks.value()
        order = self.ui.SliderOfOrder.value()
        self.chunkplotter.clear()
        self.chunkplotter.plot(
            self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal, pen="k")

        if self.numberofChunkies <= 0:
            return

        n_chunks = min(self.numberofChunkies, len(self.xAxisValuesOfOriginalSignal))
        fits = fitting.fit_chunks(
            self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal, n_chunks, order)

        x_sym = symbols("x")
        for chunkidx, f in enumerate(fits):
            xAxischunks = np.linspace(f.x_min, f.x_max, 50)
            yAxischunks = f.predict(xAxischunks)
            self.chunkplotter.plot(xAxischunks, yAxischunks, symbol='o')

            # highest-degree-first coefficients -> ascending for the latex sum
            ascending = f.coeffs[::-1]
            poly = sum(S("{:6.2f}".format(v)) * x_sym ** i for i, v in enumerate(ascending))
            eq_latex = printing.latex(poly)

            tableData['error'].append(round(float(f.rmse), 4))
            tableData['equation'].append(f'${eq_latex}$')
            tableData['Chunk'].append(f'eq of chunk {chunkidx + 1}')

        self.latexWriter.axes.clear()
        if order > 0:
            frame = pd.DataFrame(tableData)
            table = self.latexWriter.axes.table(
                cellText=frame.values, colLabels=frame.columns, loc='center')
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
