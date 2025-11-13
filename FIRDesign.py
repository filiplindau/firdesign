"""
GUI for designing FIR filters and examining attenuation of frequencies

:created: 2025-11-12
:author: Filip Lindau
"""

import sys
import logging
import threading
import time
import copy
from contextlib import contextmanager
from dataclasses import dataclass, field, astuple, fields
from collections import OrderedDict
from typing import Any
from enum import Enum, Flag
from PyQt5 import QtCore, QtWidgets
import numpy as np
import argparse
import pyqtgraph as pq
import scipy.signal as ss
from streamlit.runtime import get_instance

from fir_design_gui import Ui_FIRDesign
import multiprocessing as mp
from typing import List
from pathlib import Path

# Define a custom logging format
# log_format = "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
log_format = "%(asctime)s - %(levelname)s - %(name)s %(funcName)s - %(message)s"

# Set up the logging configuration using basicConfig
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@contextmanager
def block_signals(widget: QtWidgets.QWidget):
    widget.blockSignals(True)
    try:
        yield
    finally:
        widget.blockSignals(False)


def prefix_format(value, decimals=2, unit=""):
    e = np.log10(value) // 3
    ind_max = 5
    ind_min = -5
    ind = int(max(ind_min, min(e, ind_max)) - ind_min)
    prefix_list = ["f", "p", "n", "u", "m", "", "k", "M", "G", "P", "E"]
    s = f"{value * 10**(-3 * e):.{decimals}f} {prefix_list[ind]}{unit}"
    return s


class FilterType(Enum):
    LOWPASS = 0
    HIGHPASS = 1
    BANDPASS = 2
    NONE = 3


@dataclass
class FilterParameter:
    name: str
    value: Any = None
    min: Any = None
    max: Any = None


@dataclass
class FIRFilter:
    name: str = ""
    coe: np.array = field(default_factory=lambda: np.array([]))
    n_taps: int = 0
    type: FilterType = FilterType.NONE
    algo: str = ""
    parameters: OrderedDict[str, FilterParameter] = field(default_factory=OrderedDict)

    def __str__(self):
        return f"{self.name}: {self.n_taps} taps"

    def generate(self):
        logger.warning(f"No generator for algorithm {self.algo}")
        return None

    @staticmethod
    def create(name):
        """
        Overload for all types of filter
        :param name:
        :return:
        """
        filter = FIRFilter(name)
        return filter


@dataclass
class FirwinFilter(FIRFilter):
    @staticmethod
    def create(name, n_taps=1, type=FilterType.LOWPASS, fc=0.5, width=0.1):
        params = OrderedDict([("fc", FilterParameter("fc", fc, 0, 1)), ("width", FilterParameter("width", width, 0, 1))])
        filter  = FirwinFilter(name, n_taps=n_taps, type=type, algo="firwin", parameters=params)
        return filter

    def generate(self):
        self.coe = ss.firwin(self.n_taps, self.parameters["fc"].value, width=self.parameters["width"].value)
        return self.coe

    def __str__(self):
        return (f"{self.name}: {self.type.name} of {self.algo}, {self.n_taps} taps, fc {self.parameters['fc'].value:.3f}, "
                f"width {self.parameters['width'].value:.3f}")


@dataclass
class KaiserFilter(FIRFilter):
    @staticmethod
    def create(name, type=FilterType.LOWPASS, fc=0.5, width=0.1, stopband_att=50):
        params = OrderedDict([("fc", FilterParameter("fc", fc, 0, 1)), ("width", FilterParameter("width", width, 0, 1)), ("stopband_att", FilterParameter("stopband_att", stopband_att, 0, 100))])
        n_taps, beta = ss.kaiserord(stopband_att, width)
        filter = KaiserFilter(name, n_taps=n_taps, type=type, algo="kaiser", parameters=params)
        return filter

    def generate(self):
        n_taps, beta = ss.kaiserord(self.parameters["stopband_att"].value, self.parameters["width"].value)
        self.n_taps = n_taps
        self.coe = ss.firwin(n_taps, self.parameters["fc"].value, window=('kaiser', beta))
        logger.info(f"Generating {self.name}")
        return self.coe

    def __str__(self):
        return (f"{self.name}: {self.type.name} of {self.algo}, {self.n_taps} taps, fc {self.parameters['fc'].value:.2f}, "
                f"width {self.parameters['width'].value:.2f}, "
                f"stopband attenuation {self.parameters['stopband_att'].value:.2f} dB")


class FIRDesign(QtWidgets.QWidget):
    fft_done_signal = QtCore.pyqtSignal()
    bode_done_signal = QtCore.pyqtSignal()
    MAX_HIST = 100000

    def __init__(self, use_small=False, parent=None):
        logger.debug("Init")
        QtWidgets.QWidget.__init__(self, parent)
        self.settings = QtCore.QSettings('Maxlab', 'TF_Bode_RP')

        # Matplotlib original colors
        self.color_dict = {"c0": "#1f77b4", "c1": "#ff740e", "c2": "#2ca02c", "c3": "#d62728", "c4": "#9467bd",
                           "c5": "#8c564b", "c6": "#1f77b4", "c7": "#ff740e", "c8": "#2ca02c", "c9": "#d62728",
                           "c10": "#9467bd", "c11": "#8c564b",
                           "grey": "#7f7f7f", "white": "#eeeeee"}

        # Matplotlib dark background colors
        self.color_dict = {"c0": "#8dd3c7", "c1": "#feff8a", "c2": "#bfbbd9", "c3": "#fa8174", "c4": "#81b1d2",
                           "c5": "#fdb462", "c6": "#b3de69", "c7": "#bc82bd", "c8": "#ccebc4", "c9": "#ffed6f",
                           "c10": "#9467bd", "c11": "#8c564b",
                           "grey": "#7f7f7f", "white": "#eeeeee"}

        # Goggles colors
        self.color_dict = {"c0": "#839AEB", "c1": "#feff8a", "c2": "#BC82D9", "c3": "#fa8174", "c4": "#6DD175",
                           "c5": "#fdb462", "c6": "#b3de69", "c7": "#7C519C", "c8": "#ccebc4", "c9": "#ffed6f",
                           "c10": "#9467bd", "c11": "#8c564b",
                           "grey": "#7f7f7f", "white": "#eeeeee"}

        self.ui = Ui_FIRDesign()
        self.ui.setupUi(self)

        self.current_filter_ind = 0
        self.filter_counter = 0
        self.n_param_widgets = 0

        self.amp_plot_list = list()

        self.setup_layout()

        filt = FirwinFilter.create("default", n_taps=19, type=FilterType.LOWPASS, fc=0.5, width=0.1)
        self.add_filter(filt)

    def setup_layout(self):
        self.ui.ntaps_spinbox.setValue(self.settings.value("ntaps", 21, type=int))
        self.ui.ntaps_spinbox.editingFinished.connect(self.spinbox_update)
        self.ui.ntaps_min_spinbox.editingFinished.connect(self.slider_limit_update)
        self.ui.ntaps_max_spinbox.editingFinished.connect(self.slider_limit_update)
        self.ui.ntaps_min_spinbox.editingFinished.connect(self.slider_limit_update)
        self.ui.ntaps_slider.valueChanged.connect(self.slider_update)
        self.ui.add_filter_button.clicked.connect(self.add_filter)
        self.ui.filtertype_combobox.addItems(["lowpass", "highpass", "decimation"])
        self.ui.filtertype_combobox.currentIndexChanged.connect(self.update_parameters)
        self.ui.algo_combobox.addItems(["firwin", "kaiser", "remez"])
        self.ui.algo_combobox.currentIndexChanged.connect(self.update_parameters)
        self.ui.coe_label.setWordWrap(True)
        self.ui.coe_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard)

        self.ui.plot0_widget.getPlotItem().showGrid(True, True)
        self.ui.plot0_widget.getPlotItem().addLegend()
        self.ui.nfreq_spinbox.setValue(self.settings.value("nfreq", 2048, type=int))
        self.ui.nfreq_spinbox.editingFinished.connect(self.update_plot)

        # Find number of parameter widgets
        regex = QtCore.QRegularExpression(r"p\d+_.*")
        self.n_param_widgets = len(self.findChildren(QtWidgets.QLabel, regex))
        logger.info(f"Found {self.n_param_widgets} parameter widgets")
        for p_ind in range(self.n_param_widgets):
            getattr(self.ui, f"p{p_ind}_label").setText("--")
            obj = getattr(self.ui, f"p{p_ind}_min_spinbox")
            obj.setEnabled(False)
            obj.editingFinished.connect(self.slider_limit_update)
            obj = getattr(self.ui, f"p{p_ind}_max_spinbox")
            obj.setEnabled(False)
            obj.editingFinished.connect(self.slider_limit_update)
            obj = getattr(self.ui, f"p{p_ind}_spinbox")
            obj.setEnabled(False)
            obj.editingFinished.connect(self.spinbox_update)
            obj = getattr(self.ui, f"p{p_ind}_slider")
            obj.setEnabled(False)
            obj.valueChanged.connect(self.slider_update)

        self.ui.filters_list.currentItemChanged.connect(self.select_filter)

    def slider_update(self, value):
        """
        Slider was changed. Calculate value and update spinbox
        :param value:
        :return:
        """
        slider = self.sender()
        name = "_".join(self.sender().objectName().split("_")[:-1])
        logger.debug(f"update slider: {name}")
        filter = self.ui.filters_list.currentItem().data(QtCore.Qt.UserRole)
        if name == "ntaps":
            max_val = self.ui.ntaps_max_spinbox.value()
            min_val = self.ui.ntaps_min_spinbox.value()
        else:
            p_ind = int(name[1:])
            max_val = getattr(self.ui, f"{name}_max_spinbox").value()
            min_val = getattr(self.ui, f"{name}_min_spinbox").value()
            # param: FilterParameter = filter.parameters.values()[p_ind]
            # max_val = param.max
            # min_val = param.min
        cal_val = min_val + (max_val - min_val) * slider.value() / 100.0
        logger.debug(f"Slider {name} rel value: {value}, actual: {cal_val}")
        spinbox = self.ui.__dict__[f"{name}_spinbox"]
        if isinstance(spinbox, QtWidgets.QSpinBox):
            spinbox.setValue(int(cal_val))
        else:
            spinbox.setValue(cal_val)
        self.spinbox_update()

    def slider_limit_update(self):
        name = "_".join(self.sender().objectName().split("_")[:-1])
        logger.info(f"update slider limits: {name}")
        max_val = getattr(self.ui, f"{name}_max_spinbox").value()
        min_val = getattr(self.ui, f"{name}_min_spinbox").value()
        spinbox = getattr(self.ui, f"{name}_spinbox").value()
        slider = getattr(self.ui, f"{name}_slider").value()
        cal_val = spinbox.value()
        slider_val = int(100 * (cal_val - min_val) / (max_val - min_val))
        with block_signals(slider):
            slider.setValue(slider_val)
        self.update_parameters()

    def spinbox_update(self):
        name = "_".join(self.sender().objectName().split("_")[:-1])
        logger.debug(f"spinbox_update {name}")
        # Update slider
        max_val = getattr(self.ui, f"{name}_max_spinbox").value()
        min_val = getattr(self.ui, f"{name}_min_spinbox").value()
        spinbox = getattr(self.ui, f"{name}_spinbox")
        slider = getattr(self.ui, f"{name}_slider")
        cal_val = spinbox.value()
        slider_val = int(100 * (cal_val - min_val) / (max_val - min_val))
        logger.debug(f"Updating slider {name} to {cal_val}")
        with block_signals(slider):
            slider.setValue(slider_val)
        self.update_parameters()

    def update_parameters(self):
        """
        Update parameters stored in FIRFilter from gui data
        :return:
        """
        item: QtWidgets.QListWidgetItem = self.ui.filters_list.currentItem()
        filter: FIRFilter = item.data(QtCore.Qt.UserRole)
        logger.info(f"update_parameters {filter} of {filter.algo}")
        n_taps = self.ui.ntaps_spinbox.value()
        algo_type = self.ui.algo_combobox.currentText()
        update_widgets = False
        if algo_type != filter.algo:
            update_widgets = True
            if algo_type == "kaiser":
                filter = KaiserFilter.create(filter.name)
            else:
                filter = FirwinFilter.create(filter.name)
        filter.n_taps = n_taps
        p_keys = list(filter.parameters.keys())
        for p_ind in range(len(filter.parameters)):
            if filter.parameters[p_keys[p_ind]].name == self.findChild(QtCore.QObject, f"p{p_ind}_label").text():
                val = self.findChild(QtCore.QObject, f"p{p_ind}_spinbox").value()
                slider_min = self.findChild(QtCore.QObject, f"p{p_ind}_min_spinbox").value()
                slider_max = self.findChild(QtCore.QObject, f"p{p_ind}_max_spinbox").value()
                filter.parameters[p_keys[p_ind]].value = val
                filter.parameters[p_keys[p_ind]].min = slider_min
                filter.parameters[p_keys[p_ind]].max = slider_max
        item.setText(str(filter))
        item.setData(QtCore.Qt.UserRole, filter)
        if update_widgets:
            self.select_filter()
            return
        self.start_design()


    def start_design(self):
        filt = self.ui.filters_list.currentItem().data(QtCore.Qt.UserRole)
        coe = filt.generate()
        coe_str = ", ".join([f"{c:.3e}" for c in coe])
        logger.info(f"Generating {filt.name} filter: {filt.n_taps} taps using {filt.algo}.")
        self.ui.coe_label.setText(coe_str)
        with block_signals(self.ui.ntaps_spinbox):
            self.ui.ntaps_spinbox.setValue(filt.n_taps)
        max_val = self.ui.ntaps_max_spinbox.value()
        min_val = self.ui.ntaps_min_spinbox.value()
        cal_val = filt.n_taps
        slider_val = int(100 * (cal_val - min_val) / (max_val - min_val))
        with block_signals(self.ui.ntaps_slider):
            self.ui.ntaps_slider.setValue(slider_val)

        self.update_plot()
        return

    def add_filter(self, filt: FIRFilter = None):
        logger.info(f"Adding filter {self.filter_counter}")
        if not isinstance(filt, FIRFilter):
            filt = copy.copy(self.ui.filters_list.currentItem().data(QtCore.Qt.UserRole))
            filt.name = f"Filter {self.filter_counter}"

        item = QtWidgets.QListWidgetItem(f"{filt.name}")
        item.setData(QtCore.Qt.UserRole, filt)
        with block_signals(self.ui.filters_list):
            self.ui.filters_list.addItem(item)
            self.ui.filters_list.setCurrentItem(item)
        plot = self.ui.plot0_widget.plot(antialias=True, name=filt.name)
        plot.setLogMode(False, False)
        color_ind = self.filter_counter % len(self.color_dict)
        color = list(self.color_dict.values())[color_ind]
        plot.setPen(pq.mkPen(color=color, width=2.0))
        self.amp_plot_list.append(plot)
        self.filter_counter += 1
        logger.info(f"Filter #{self.filter_counter} / {len(self.amp_plot_list)}")
        self.select_filter()

    def update_plot(self):
        item = self.ui.filters_list.currentItem()
        ind = self.ui.filters_list.currentRow()
        filt: FIRFilter = item.data(QtCore.Qt.UserRole)
        w, h = ss.freqz(filt.coe, worN=self.ui.nfreq_spinbox.value())
        f = w / np.pi
        logger.debug(f"Updating plot. Filter #{ind} / {len(self.amp_plot_list)}: {filt.name}")
        self.amp_plot_list[ind].setData(f, 20*np.log10(np.abs(h)))

    def select_filter(self):
        item = self.ui.filters_list.currentItem()
        filt: FIRFilter = item.data(QtCore.Qt.UserRole)
        logger.info(f"select_filter {item}: {filt}")
        slider_val = int(100 * (filt.n_taps - self.ui.ntaps_min_spinbox.value()) / (self.ui.ntaps_max_spinbox.value() - self.ui.ntaps_min_spinbox.value()))
        slider = self.ui.ntaps_slider
        slider.setEnabled(True)
        with block_signals(slider):
            slider.setValue(slider_val)
        with block_signals(self.ui.ntaps_spinbox):
            self.ui.ntaps_spinbox.setValue(filt.n_taps)
        with block_signals(self.ui.algo_combobox):
            self.ui.algo_combobox.setCurrentText(filt.algo)
        with block_signals(self.ui.filtertype_combobox):
            self.ui.filtertype_combobox.setCurrentText(filt.type.name)
        p_keys = filt.parameters.keys()
        for ind, p in enumerate(p_keys):
            logger.info(f"Enabling {filt.parameters[p]}")
            min_val = filt.parameters[p].min
            max_val = filt.parameters[p].max
            val = filt.parameters[p].value

            getattr(self.ui, f"p{ind}_label").setText(p)
            obj = getattr(self.ui, f"p{ind}_min_spinbox")
            with block_signals(obj):
                obj.setEnabled(True)
                obj.setValue(min_val)
            obj = getattr(self.ui, f"p{ind}_max_spinbox")
            with block_signals(obj):
                obj.setEnabled(True)
                obj.setValue(max_val)
            obj = getattr(self.ui, f"p{ind}_spinbox")
            with block_signals(obj):
                obj.setEnabled(True)
                obj.setValue(val)
            slider_val = int(100 * (val - min_val) / (max_val - min_val))
            logger.info(f"Updating slider {ind} to {val}")
            slider = getattr(self.ui, f"p{ind}_slider")
            slider.setEnabled(True)
            with block_signals(slider):
                slider.setValue(slider_val)

        for p_ind in range(len(p_keys), self.n_param_widgets):
            getattr(self.ui, f"p{p_ind}_label").setText("--")
            getattr(self.ui, f"p{p_ind}_min_spinbox").setEnabled(False)
            getattr(self.ui, f"p{p_ind}_max_spinbox").setEnabled(False)
            getattr(self.ui, f"p{p_ind}_spinbox").setEnabled(False)
            getattr(self.ui, f"p{p_ind}_slider").setEnabled(False)

        self.start_design()

    def closeEvent(self, a0, QCloseEvent=None):
        logger.info(f"Saving settings.")
        self.settings.setValue("ntaps", self.ui.ntaps_spinbox.value())
        self.settings.setValue("nfreq", self.ui.nfreq_spinbox.value())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-gl", "--use_opengl", action="store_true",
                        help="Set to use opengl rendering in pyqtgraph.")

    args = parser.parse_args()
    if args.use_opengl:
        pq.setConfigOptions(useOpenGL=True, useNumba=True)
    else:
        pq.setConfigOptions(useOpenGL=False, useNumba=True)

    np.seterr(divide='ignore', invalid='ignore', over='ignore')

    # Enable High DPI scaling
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

    # Use High DPI Pixmaps (optional, but usually looks better)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    myapp = FIRDesign()
    myapp.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()