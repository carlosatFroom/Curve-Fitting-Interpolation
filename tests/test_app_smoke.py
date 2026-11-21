import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt5 import QtCore, QtWidgets  # noqa: E402
from PyQt5.QtWidgets import QFileDialog  # noqa: E402

import app as appmod  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def sine(qapp):
    data = np.genfromtxt(ROOT / "samples" / "sine.csv", delimiter=",")[:1000]
    return data[:, 0], data[:, 1]


@pytest.fixture
def window(qapp, sine):
    w = appmod.mainApp()
    w.setSignal(*sine)
    yield w
    w.close()


def test_chunks_draws_fits_and_table(window):
    window.ui.SliderOfOrder.setValue(3)
    window.ui.SliderOfChunks.setValue(4)
    window.chunks()
    # data curve + one fitted curve per chunk
    assert len(window.chunkplotter.listDataItems()) == 1 + 4
    assert len(window.latexWriter.axes.tables) == 1
    table = window.latexWriter.axes.tables[0]
    assert len({row for row, _ in table.get_celld()}) == 1 + 4  # header + chunks


def test_chunks_without_data_does_not_raise(qapp):
    w = appmod.mainApp()
    w.chunks()
    w.close()


def test_chunks_zero_clears_fits(window):
    window.ui.SliderOfChunks.setValue(2)
    assert len(window.chunkplotter.listDataItems()) == 1 + 2
    window.ui.SliderOfChunks.setValue(0)
    assert len(window.chunkplotter.listDataItems()) == 1  # data curve only


def test_extrapolation_toggle(window, sine):
    x, _ = sine
    window.ui.SliderOfOrder.setValue(3)
    window.ui.SliderOfChunks.setValue(5)
    window.ui.extrapolationCheckBox.setChecked(True)
    assert window.verticalLinePlotter.getData()[0][0] == pytest.approx(x[499])  # 50% of 1000 points
    window.ui.SliderOfChunks.setValue(0)
    assert window.verticalLinePlotter.getData()[0][0] == pytest.approx(x[3])  # order+1 points
    window.ui.extrapolationCheckBox.setChecked(False)
    window.ui.SliderOfChunks.setValue(2)
    assert len(window.chunkplotter.listDataItems()) == 1 + 2


def test_error_maps_compute_and_display(window, sine):
    x, y = sine
    maps = appmod.compute_error_maps(x, y, 5, 5, fixed_order=5)
    assert maps['order_chunks'].shape == (5, 5)
    assert maps['chunks_overlap'].shape == (5, len(appmod.OVERLAPS))
    assert maps['order_overlap'].shape == (5, len(appmod.OVERLAPS))
    window.onErrorMapReady(maps)
    assert window.errorMapGenerated
    for key in appmod.MAP_AXES:
        assert window.errorMaps[key].max() == pytest.approx(100.0)

    # transposed view comes from the same normalised data
    assert np.allclose(window._errorMapFor(0, 1), window._errorMapFor(1, 0).T)
    assert window._errorMapFor(0, 1).shape == (5, 5)
    assert window._errorMapFor(2, 0).shape == (5, len(appmod.OVERLAPS))  # rows = chunks

    window.errorMapCanv.setVisible(True)
    for x_idx in range(3):
        for y_idx in range(3):
            if x_idx == y_idx:
                continue
            window.ui.xAxesComboBox.setCurrentIndex(x_idx)
            window.ui.yAxesComboBox.setCurrentIndex(y_idx)
            window.showErrorMap()
            assert len(window.errorMapCanv.axes.images) == 1


def test_error_map_cancelled_is_none(window, sine):
    x, y = sine
    assert appmod.compute_error_maps(x, y, 3, 3, 5, is_cancelled=lambda: True) is None
    window.onErrorMapReady(None)  # must not raise


def test_error_map_worker_emits_result(qapp, sine):
    x, y = sine
    received = []
    worker = appmod.ErrorMapWorker(x, y, 3, 3, 5)
    worker.resultReady.connect(received.append)
    worker.start()
    assert worker.wait(20000)
    QtCore.QCoreApplication.processEvents()
    assert received and received[0]['order_chunks'].shape == (3, 3)


def test_read_file_cancelled_dialog_keeps_data(window, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", "")))
    before = window.yAxisValuesOfOriginalSignal.copy()
    window.read_file()
    assert np.array_equal(window.yAxisValuesOfOriginalSignal, before)
