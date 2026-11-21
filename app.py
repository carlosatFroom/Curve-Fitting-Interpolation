import os
import sys
import pyqtgraph as pg, matplotlib, numpy as np, pathlib
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QSlider
from pyqtgraph import *
from sympy import S, symbols, printing
from mainapp import Ui_MainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

import fitting

matplotlib.use('QT5Agg')


class ErrorMapWorker(QThread):
    """Computes the three error maps off the GUI thread. No drawing here."""

    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, x, y, maxOrder, maxChunks, parent=None):
        super(ErrorMapWorker, self).__init__(parent)
        self.x = x
        self.y = y
        self.maxOrder = maxOrder
        self.maxChunks = maxChunks
        self.cancelled = False

    def run(self):
        overlaps = range(0, 26, 5)
        try:
            results = {}
            # order x chunks, overlap 0
            results['orderChunks'] = fitting.error_map(
                self.x, self.y, range(1, self.maxOrder + 1), range(1, self.maxChunks + 1), [0])
            if self.cancelled:
                return
            # chunks x overlap, order fixed at 5 -> transpose to (overlap, chunks)
            results['chunksOverlap'] = fitting.error_map(
                self.x, self.y, [5], range(1, self.maxChunks + 1), overlaps).T
            if self.cancelled:
                return
            # order x overlap, chunks fixed at 5
            results['orderOverlap'] = fitting.error_map(
                self.x, self.y, range(1, self.maxOrder + 1), [5], overlaps)
        except ValueError as exc:
            self.failed.emit(str(exc))
            return
        if not self.cancelled:
            self.done.emit(results)


class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self,parent=None, dpi = 120):
        self.fig, self.axes = plt.subplots()
        super(MatplotlibCanvas,self).__init__(self.fig)
        self.axes.set_facecolor((242/255,243/255,245/255))
        self.fig.set_facecolor((242/255,243/255,245/255))


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
        self.errorMapWorker = None
        divider = make_axes_locatable(self.errorMapCanv.axes)
        self.cax = divider.append_axes('right', size='5%', pad=0.05)
        self.cax.set_facecolor((242/255,243/255,245/255))


    def errorMapStart(self):
        if self.errorMapWorker is not None and self.errorMapWorker.isRunning():
            return
        self.maxPolyNomialOrder = self.ui.orderOfFittingPolynomialErrorMapSlider.value()
        self.maxNumberOfChunks = self.ui.numberOfChunksErrorMapSlider.value()
        self.errorMapWorker = ErrorMapWorker(
            self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal,
            self.maxPolyNomialOrder, self.maxNumberOfChunks)
        self.errorMapWorker.done.connect(self.onErrorMapReady)
        self.errorMapWorker.failed.connect(self.onErrorMapFailed)
        self.errorMapWorker.start()

    def errorMapCancel(self):
        if self.errorMapWorker is not None:
            self.errorMapWorker.cancelled = True

    def onErrorMapReady(self, results):
        # runs on the GUI thread (queued signal from the worker)
        [self.polynomialOrderToNumberOfChunksError, self.numberOfChunksToPolynomialOrderError] = self.normalizeAndPrepData(results['orderChunks'])
        [self.overLappingPercentageToNumberOfChunksError, self.numberOfChunksToOverLappingPercentageError] = self.normalizeAndPrepData(results['chunksOverlap'])
        [self.polynomialOrderToOverlappingPercentageError, self.overLappingPercentageToPolynomialOrderError] = self.normalizeAndPrepData(results['orderOverlap'])
        self.errorMapGenerated = True
        self.showErrorMap()

    def onErrorMapFailed(self, message):
        QMessageBox.warning(self, "Error map", message)


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
        x = self.xAxisValuesOfOriginalSignal
        y = self.yAxisValuesOfOriginalSignal
        fraction = self.ui.SliderOfChunks.value() / 10
        if fraction <= 0:
            self.verticalLinePlotter.setData([],[])
            self.extraPolationPlotter.setData([],[])
            return
        polynomialOrder = self.ui.SliderOfOrder.value()
        _, yPred = fitting.extrapolate(x, y, fraction, polynomialOrder)
        verticalLineXIndex = x[max(int(len(x) * fraction), 1) - 1]
        self.verticalLinePlotter.setData([verticalLineXIndex,verticalLineXIndex],[min(y),max(y)])
        self.extraPolationPlotter.setData(x, yPred)

    def showErrorMap(self):
        if self.errorMapGenerated == False:
            return

        self.errorMapCanv.axes.clear()
        self.cax.clear()

        xAxesComboBoxIndex = self.ui.xAxesComboBox.currentIndex()
        yAxesComboBoxIndex = self.ui.yAxesComboBox.currentIndex()

        if (xAxesComboBoxIndex == 0 and yAxesComboBoxIndex == 1):
            image = self.errorMapCanv.axes.imshow(self.polynomialOrderToNumberOfChunksError,cmap = "inferno",extent=[0,self.maxNumberOfChunks,0,self.maxPolyNomialOrder])
            self.errorMapCanv.axes.set_xlabel("number of chunks")
            self.errorMapCanv.axes.set_ylabel("polynomial order")
        elif (xAxesComboBoxIndex == 1 and yAxesComboBoxIndex == 0):
            image = self.errorMapCanv.axes.imshow(self.numberOfChunksToPolynomialOrderError,cmap = "inferno",extent=[0,self.maxPolyNomialOrder,0,self.maxNumberOfChunks])
            self.errorMapCanv.axes.set_xlabel("polynomial order")
            self.errorMapCanv.axes.set_ylabel("number of chunks")

        elif (xAxesComboBoxIndex == 0 and yAxesComboBoxIndex == 2):
            image = self.errorMapCanv.axes.imshow(self.overLappingPercentageToNumberOfChunksError,cmap = "inferno",extent=[0,self.maxNumberOfChunks,0,25])
            self.errorMapCanv.axes.set_xlabel("number of chunks")
            self.errorMapCanv.axes.set_ylabel("overlapping percentage")
        elif (xAxesComboBoxIndex == 2 and yAxesComboBoxIndex == 0):
            image = self.errorMapCanv.axes.imshow(self.numberOfChunksToOverLappingPercentageError,cmap = "inferno",extent=[0,25,0,self.maxNumberOfChunks])
            self.errorMapCanv.axes.set_xlabel("overlapping percentage")
            self.errorMapCanv.axes.set_ylabel("number of chunks")

        elif (xAxesComboBoxIndex == 1 and yAxesComboBoxIndex == 2):
            image = self.errorMapCanv.axes.imshow(self.overLappingPercentageToPolynomialOrderError,cmap = "inferno",extent=[0,self.maxPolyNomialOrder,0,25])
            self.errorMapCanv.axes.set_xlabel("polynomial order")
            self.errorMapCanv.axes.set_ylabel("overlapping percentage")
        else:
            image = self.errorMapCanv.axes.imshow(self.polynomialOrderToOverlappingPercentageError,cmap = "inferno",extent=[0,25,0,self.maxPolyNomialOrder])
            self.errorMapCanv.axes.set_xlabel("overlapping percentage")
            self.errorMapCanv.axes.set_ylabel("polynomial order")


        self.errorMapCanv.fig.colorbar(image, cax=self.cax, orientation='vertical')
        self.errorMapCanv.fig.tight_layout()
        self.errorMapCanv.draw()

    def normalizeAndPrepData(self,data):
        swapedAxesErrorData = np.flip(np.transpose(data),0)
        errorData = np.flip(data,0)
        return [errorData, swapedAxesErrorData]

    def read_file(self):

        path = QFileDialog.getOpenFileName()[0]
        self.FileName= os.path.basename(path)

        if pathlib.Path(path).suffix == ".csv":
            self.data = np.genfromtxt(path, delimiter=',')
            self.xAxisValuesOfOriginalSignal = self.data[:,0][:1000]
            self.yAxisValuesOfOriginalSignal = self.data[:,1][:1000]
            self.mainSignalDataPlotter.setData(self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal)

        self.ui.mainSignal.setLimits(yMin = min(self.yAxisValuesOfOriginalSignal), yMax = max(self.yAxisValuesOfOriginalSignal))

    def chunks(self):
        tableData = {'Chunk': [], 'equation': [], 'RMSE': []}
        self.numberofChunkies = self.ui.SliderOfChunks.value()
        self.chunkplotter.clear()
        self.chunkplotter.plot(self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal,pen = "k")
        self.i=0
        order = self.ui.SliderOfOrder.value()
        if(self.numberofChunkies > 0):
            fits = fitting.fit_chunks(self.xAxisValuesOfOriginalSignal, self.yAxisValuesOfOriginalSignal, self.numberofChunkies, order)
            xSymbol = symbols("x")

            for chunkidx, fit in enumerate(fits):
                xs = np.linspace(fit.x_min, fit.x_max)
                ys = fit.predict(xs)

                poly = sum(S("{:6.2f}".format(v)) * xSymbol ** i
                                                for i, v in enumerate(fit.coeffs[::-1]))
                eq_latex = printing.latex(poly)

                self.chunkplotter.plot(xs, ys, symbol='o')
                tableData['RMSE'].append(round(fit.rmse, 3))
                tableData['equation'].append(f'${eq_latex}$')
                tableData['Chunk'].append(f'eq of chunk {(chunkidx+1)}')

            self.latexWriter.axes.clear()

            if (order > 0): ## plot the table only if the spin box has value
                TableData_frame = pd.DataFrame(tableData) ## forming the table with pandas library
                table=self.latexWriter.axes.table(cellText=TableData_frame.values, colLabels=TableData_frame.columns, loc='center')
                table.auto_set_column_width(col=list(range(len(TableData_frame.columns)))) ## orientation
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
