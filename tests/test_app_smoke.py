import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = r"""
import numpy as np
from PyQt5 import QtCore, QtWidgets
import app

qt = QtWidgets.QApplication([])
w = app.mainApp()
d = np.genfromtxt('samples/ECG2.csv', delimiter=',')[:1000]
w.xAxisValuesOfOriginalSignal, w.yAxisValuesOfOriginalSignal = d[:, 0], d[:, 1]
w.ui.SliderOfOrder.setValue(3)
w.ui.SliderOfChunks.setValue(4)          # triggers chunks()
w.ui.extrapolationCheckBox.setChecked(True)
w.ui.SliderOfChunks.setValue(5)          # triggers updateExtraPolation()
w.ui.extrapolationCheckBox.setChecked(False)

w.ui.orderOfFittingPolynomialErrorMapSlider.setValue(4)
w.ui.numberOfChunksErrorMapSlider.setValue(3)
loop = QtCore.QEventLoop()
w.errorMapStart()
w.errorMapThread.finished.connect(loop.quit)
QtCore.QTimer.singleShot(20000, loop.quit)
loop.exec_()
qt.processEvents()
assert w.errorMaps is not None
assert w.errorMaps['order_chunks'].shape == (4, 3)
for xi in range(3):
    for yi in range(3):
        w.ui.xAxesComboBox.setCurrentIndex(xi)
        w.ui.yAxesComboBox.setCurrentIndex(yi)
        w.showErrorMap()
print('OK')
"""


def test_app_runs_headless():
    pytest.importorskip("PyQt5")
    pytest.importorskip("pyqtgraph")
    pytest.importorskip("matplotlib")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    r = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr


def test_poly_latex():
    pytest.importorskip("PyQt5")
    pytest.importorskip("matplotlib")
    sys.path.insert(0, ROOT)
    from app import poly_latex
    assert poly_latex([1.0, -2.0, 0.0, 3.5]) == "1.00 x^{3} - 2.00 x^{2} + 3.50"
    assert poly_latex([-1.0, 0.0]) == "-1.00 x"
    assert poly_latex([0.0]) == "0"
