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
from dataclasses import dataclass, field, astuple, fields, asdict
from collections import OrderedDict
from typing import Any, ClassVar, Dict, Type
from enum import Enum, Flag

import pyqtgraph
from PyQt5 import QtCore, QtWidgets, QtGui
import numpy as np
import argparse
import pyqtgraph as pq
import scipy.signal as ss
from scipy.interpolate import interp1d
#from streamlit.runtime import get_instance
#from zope.interface import named

from fir_design_gui import Ui_FIRDesign
import multiprocessing as mp
from typing import List
import pathlib
import yaml

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

    def get_dict(self):
        d = dict()
        d["name"] = self.name
        d["value"] = self.value
        d["min"] = self.min
        d["max"] = self.max
        return d


@dataclass
class FIRFilter:
    name: str = ""
    coe: np.array = field(default_factory=lambda: np.array([]))
    n_taps: int = 0
    type: FilterType = FilterType.NONE
    algo: str = ""
    decimation: int = 1
    parameters: OrderedDict[str, FilterParameter] = field(default_factory=OrderedDict)

    # class-level registry mapping type name -> subclass
    _registry: ClassVar[Dict[str, Type["FirFilter"]]] = {}

    @classmethod
    def register_subclass(cls, name: str):
        """Decorator: register a subclass under 'name'."""
        def decorator(subcls):
            cls._registry[name] = subcls
            subcls._type_name = name
            return subcls
        return decorator

    def __str__(self):
        return f"{self.name}: {self.n_taps} taps. {self.decimation}x decimation"

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

    def to_yaml(self):
        d = self.to_dict()
        return yaml.safe_dump(d, sort_keys=False)

    def to_dict(self):
        d = dict()
        d["name"] = self.name
        d["type"] = self.type.name
        d["algo"] = self.algo
        d["n_taps"] = self.n_taps
        d["decimation"] = self.decimation
        params = [p.get_dict() for p in self.parameters.values()]
        d["parameters"] = params
        d["coe"] = ", ".join([f"{x:.6f}" for x in self.coe.tolist()])
        return d

    @classmethod
    def from_yaml(cls, data):
        # d = yaml.safe_load(data)
        d = data
        if "algo" not in d:
            raise KeyError("Missing 'type' discriminator in serialized filter.")
        t = d["algo"]
        subcls: FIRFilter = cls._registry.get(t)
        if subcls is None:
            # as fallback, try by classname:
            subcls = cls._registry.get(t) or next(
                (c for n, c in cls._registry.items() if n == t), None
            )
        if subcls is None:
            raise ValueError(f"Unknown filter type '{t}'. Registered: {list(cls._registry)}")
        args = dict()
        args["name"] = d["name"]
        args["type"] = FilterType[d["type"]]
        args["n_taps"] = d["n_taps"]
        args["decimation"] = d["decimation"]
        args["algo"] = d["algo"]
        args["parameters"] = {fp["name"]: FilterParameter(**fp) for fp in d["parameters"]}
        args["coe"] = np.array([float(x) for x in d["coe"].split(",")])
        filter = subcls(**args)
        return filter


@FIRFilter.register_subclass("firwin")
@dataclass
class FirwinFilter(FIRFilter):
    @staticmethod
    def create(name, n_taps=1, type=FilterType.LOWPASS, fc=0.5, width=0.1, decimation=1):
        params = OrderedDict([("fc", FilterParameter("fc", fc, 0, 1)), ("width", FilterParameter("width", width, 0, 1))])
        filter  = FirwinFilter(name, n_taps=n_taps, type=type, algo="firwin", parameters=params)
        return filter

    def generate(self):
        self.coe = ss.firwin(self.n_taps, self.parameters["fc"].value, width=self.parameters["width"].value)
        return self.coe

    def __str__(self):
        return (f"{self.name}: {self.type.name} of {self.algo}, {self.n_taps} taps, fc {self.parameters['fc'].value:.3f}, "
                f"width {self.parameters['width'].value:.3f}, {self.decimation}x decimation")


@FIRFilter.register_subclass("kaiser")
@dataclass
class KaiserFilter(FIRFilter):
    @staticmethod
    def create(name, type=FilterType.LOWPASS, fc=0.5, width=0.1, stopband_att=50, decimation=1):
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
                f"stopband attenuation {self.parameters['stopband_att'].value:.2f} dB, {self.decimation}x decimation")


@FIRFilter.register_subclass("remez")
@dataclass
class RemezFilter(FIRFilter):
    @staticmethod
    def create(name, n_taps=1, type=FilterType.LOWPASS, fc=0.5, width=0.1, decimation=1):
        params = OrderedDict([("fc", FilterParameter("fc", fc, 0, 1)), ("width", FilterParameter("width", width, 0, 1))])
        filter  = RemezFilter(name, n_taps=n_taps, type=type, algo="remez", parameters=params)
        return filter

    def generate(self):
        fc = self.parameters["fc"].value
        width = self.parameters["width"].value
        bands = [0, fc - width/2, fc + width/2, 1]
        self.coe = ss.remez(self.n_taps, bands, desired=[1, 0], fs=2)
        return self.coe

    def __str__(self):
        return (f"{self.name}: {self.type.name} of {self.algo}, {self.n_taps} taps, fc {self.parameters['fc'].value:.3f}, "
                f"width {self.parameters['width'].value:.3f}, {self.decimation}x decimation")

NYQUIST = 48000


class FrequencyModel(QtCore.QAbstractTableModel):
    COL_NAME = 0
    COL_COLOR = 1
    COL_FREQ = 2
    COL_ALIAS = 3
    COL_AMP = 4
    COL_FILTERED = 5

    headers = ["Name", "", "Freq", "Aliased", "Amp / dB", "Filtered / dB"]

    def __init__(self, data=None, filter_calc_cb=None, alias_calc_cb=None):
        super().__init__()
        self._data = data or []   # list of dicts
        self.suffix_list = ["Hz", "kHz", "MHz", "GHz", "THz", "PHz"]
        self.suffix_str = ""
        self.suffix_factor = 1
        self.filter_calc_cb = filter_calc_cb
        self.alias_calc_cb = alias_calc_cb

    # ---------------------------
    # Required model methods
    # ---------------------------
    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return 6

    def data(self, index, role):
        if not index.isValid():
            return QtCore.QVariant()

        row = index.row()
        col = index.column()
        item = self._data[row]

        # Display role
        if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
            if col == self.COL_NAME:
                return item.get("name", "")

            if col == self.COL_COLOR:
                return ""

            if col == self.COL_FREQ:
                f = item.get("freq", 0) * self.suffix_factor
                return f"{f:.2f}"

            if col == self.COL_ALIAS:
                try:
                    f = self.alias_calc_cb(item['freq']) * self.suffix_factor
                    return f"{f:.2f}"
                except TypeError as e:
                    logger.exception(f"alias")
                    return "--"

            if col == self.COL_AMP:
                return item.get("amp", 0)

            if col == self.COL_FILTERED:
                try:
                    f = item['freq']
                    filt_amp = self.filter_calc_cb(f, item['amp'])
                    return f"{filt_amp:.2f}"
                except TypeError as e:
                    logger.exception(f"col amp filt")
                    return "--"
        elif role == QtCore.Qt.UserRole:
            if col == self.COL_NAME:
                return item.get("name", "")

            if col == self.COL_COLOR:
                return QtGui.QColor.toRgb(item.get("color", 0))

            if col == self.COL_FREQ:
                f = item.get("freq", 0)
                return f

            if col == self.COL_ALIAS:
                try:
                    f = self.alias_calc_cb(item['freq'])
                    return f
                except TypeError as e:
                    logger.exception(f"alias")
                    return 0

            if col == self.COL_AMP:
                return item.get("amp", 0)

            if col == self.COL_FILTERED:
                try:
                    # f = self.alias_calc_cb(item['freq'])
                    f = item['freq']
                    filt_amp = self.filter_calc_cb(f, item['amp'])
                    return filt_amp
                except TypeError as e:
                    logger.exception(f"col amp filt")
                    return 0
        elif role == QtCore.Qt.DecorationRole:
            if col == self.COL_COLOR:
                color = item.get("color", QtGui.QColor(0x333333))
                return color

        return QtCore.QVariant()

    def headerData(self, section, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            header_s = self.headers[section]
            if section in [self.COL_FREQ, self.COL_ALIAS]:
                header_s += f" / {self.suffix_str}"
            return header_s
        return QtCore.QVariant()

    # ---------------------------
    # Editing
    # ---------------------------
    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.NoItemFlags

        col = index.column()

        if col in (self.COL_NAME, self.COL_FREQ, self.COL_AMP):
            return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable

        return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled

    def setData(self, index, value, role):
        if role != QtCore.Qt.EditRole:
            return False

        row = index.row()
        col = index.column()

        try:
            if col == self.COL_NAME:
                self._data[row]["name"] = str(value)

            elif col == self.COL_COLOR:
                self._data[row]["color"] = QtGui.QColor(value)

            elif col == self.COL_FREQ:
                self._data[row]["freq"] = float(value) / self.suffix_factor

            elif col == self.COL_AMP:
                self._data[row]["amp"] = float(value)
            else:
                return False
        except ValueError:
            return False

        # Notify views that data changed (also derived columns)
        self.dataChanged.emit(
            self.index(row, 0),
            self.index(row, self.columnCount() - 1)
        )
        return True

    # ---------------------------
    # Derived columns
    # ---------------------------


    def update_filtered_amplitudes(self):
        # Notify the view that the column has changed
        top_left = self.index(0, self.COL_FILTERED)
        bottom_right = self.index(len(self._data) - 1, self.COL_FILTERED)
        self.dataChanged.emit(top_left, bottom_right)

    def update_alias_freq(self):
        # Notify the view that the column has changed
        top_left = self.index(0, self.COL_ALIAS)
        bottom_right = self.index(len(self._data) - 1, self.COL_ALIAS)
        self.dataChanged.emit(top_left, bottom_right)

    def update_suffix(self, suffix_factor):
        # for r in range(len(self._data)):
        #     k = self.suffix_factor / suffix_factor
        #     self._data[r]["freq"] = self._data[r]["freq"] * k
        ind = int(np.log10(suffix_factor) // 3)
        self.suffix_factor = 10**(-3 * ind)
        self.suffix_str = self.suffix_list[ind]
        logger.info(f"New suffix for {suffix_factor}: {self.suffix_str}, factor {self.suffix_factor:.2e}")
        self.headerDataChanged.emit(QtCore.Qt.Horizontal, self.COL_FREQ, self.COL_ALIAS)
        top_left = self.index(0, self.COL_FREQ)
        bottom_right = self.index(len(self._data) - 1, self.COL_ALIAS)
        self.dataChanged.emit(top_left, bottom_right)

    # ---------------------------
    # Add/remove rows
    # ---------------------------
    def addRow(self, name="", freq=0.0, amp=0.0, color="#333333"):
        self.beginInsertRows(QtCore.QModelIndex(), len(self._data), len(self._data))
        self._data.append({"name": name, "freq": freq, "amp": amp, "color": QtGui.QColor(color)})
        self.endInsertRows()

    def removeRow(self, row):
        if 0 <= row < len(self._data):
            self.beginRemoveRows(QtCore.QModelIndex(), row, row)
            del self._data[row]
            self.endRemoveRows()



class FIRDesign(QtWidgets.QWidget):

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

        self.highlight_colors = [
            "#FF00FF",  # neon magenta
            "#00FFFF",  # cyan / aqua
            "#FF1493",  # deep pink
            "#00FF00",  # pure neon green
            "#FFA500",  # bright orange
            "#00BFFF",  # deep sky blue
            "#FFD700",  # gold (brighter than yellow)
        ]

        self.highlight_colors = [
            "#F4A7B9",  # soft raspberry
            "#F5C08A",  # warm apricot
            "#F0E38A",  # pastel golden yellow
            "#C6E68A",  # apple green
            "#8FD9A7",  # mint teal
            "#8DD6E8",  # ice blue
            "#A6B9F0",  # soft cornflower
            "#C9A6F0",  # lavender purple
            "#E0A6D9",  # orchid pink
            "#D6C3A6",  # sand beige
            "#A6D6C3",  # pale turquoise
            "#F0B6A6",  # coral peach
        ]

        self.ui = Ui_FIRDesign()
        self.ui.setupUi(self)

        self.last_load_dir: pathlib.Path = None

        self.current_filter_ind = 0
        self.filter_counter = 0
        self.freq_counter = 0
        self.n_param_widgets = 0

        self.amp_plot_list = list()
        self.ph_plot_list = list()
        self.dec_plot_list = list()
        self.freq_scatterplot = None

        self.model = FrequencyModel(filter_calc_cb=self.calc_freq_response, alias_calc_cb=self.calc_alias_freq)

        self.setup_layout()

        filt = FirwinFilter.create("default", n_taps=19, type=FilterType.LOWPASS, fc=0.5, width=0.1, decimation=1)
        self.add_filter(filt)

    def setup_layout(self):
        self.ui.ntaps_spinbox.setValue(self.settings.value("ntaps", 21, type=int))
        self.ui.ntaps_spinbox.editingFinished.connect(self.spinbox_update)
        self.ui.ntaps_min_spinbox.editingFinished.connect(self.slider_limit_update)
        self.ui.ntaps_max_spinbox.editingFinished.connect(self.slider_limit_update)
        self.ui.ntaps_min_spinbox.editingFinished.connect(self.slider_limit_update)
        self.ui.ntaps_slider.valueChanged.connect(self.slider_update)
        self.ui.decimation_spinbox.valueChanged.connect(self.update_parameters)
        self.ui.add_filter_button.clicked.connect(self.add_filter)
        self.ui.remove_filter_button.clicked.connect(self.remove_filter)
        self.ui.filtertype_combobox.addItems(["lowpass", "highpass", "decimation"])
        self.ui.filtertype_combobox.currentIndexChanged.connect(self.update_parameters)
        self.ui.algo_combobox.addItems(["firwin", "kaiser", "remez"])
        self.ui.algo_combobox.currentIndexChanged.connect(self.update_parameters)
        self.ui.coe_label.setWordWrap(True)
        self.ui.coe_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard)

        self.ui.plot0_widget.getPlotItem().showGrid(True, True)
        self.ui.plot0_widget.getPlotItem().addLegend()
        self.ui.plot0_widget.setYRange(-5, -105)
        self.ui.plot2_widget.getPlotItem().showGrid(True, True)
        self.ui.plot1_widget.getPlotItem().showGrid(True, True)
        self.ui.plot1_widget.setYRange(-5, -105)
        ax = self.ui.plot1_widget.getPlotItem().getAxis("bottom")
        ax.enableAutoSIPrefix(True)
        ax.setLabel("Freq", units="Hz")
        self.freq_scatterplot = pyqtgraph.ScatterPlotItem()
        self.ui.plot1_widget.addItem(self.freq_scatterplot)
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

        # Freq planner
        self.ui.sampling_freq_combobox.addItems(["Hz", "kHz", "MHz", "GHz"])
        self.ui.sampling_freq_combobox.currentIndexChanged.connect(self.set_freq_suffix)
        self.ui.sampling_freq_combobox.setCurrentText(self.settings.value("sampling_freq_suffix", "MHz", str))
        self.ui.sampling_freq_spinbox.setValue(self.settings.value("sampling_freq", 307, float))
        self.ui.sampling_freq_spinbox.editingFinished.connect(self.set_sampling_freq)

        self.ui.freq_table.setModel(self.model)
        self.ui.freq_table.setSelectionBehavior(self.ui.freq_table.SelectRows)
        self.ui.freq_table.setSelectionMode(self.ui.freq_table.SingleSelection)
        self.model.dataChanged.connect(self.update_freq_plot)
        self.ui.freq_table.resizeColumnsToContents()
        self.ui.add_freq_button.clicked.connect(self.add_freq)
        self.ui.remove_freq_button.clicked.connect(self.remove_freq)

        state = self.settings.value("splitter", None)
        if state is not None:
            self.ui.splitter.restoreState(state)

        state = self.settings.value("splitter2", None)
        if state is not None:
            self.ui.splitter_2.restoreState(state)

        self.last_load_dir = self.settings.value("load_dir", pathlib.Path("."))
        self.ui.save_filter_button.clicked.connect(self.save_filters)
        self.ui.load_filter_button.clicked.connect(self.load_filters)
        self.ui.save_freq_button.clicked.connect(self.save_freqs)
        self.ui.load_freq_button.clicked.connect(self.load_freqs)

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
        name = self.sender().objectName().split("_")[0]
        logger.info(f"update slider limits: {name}")
        max_val = getattr(self.ui, f"{name}_max_spinbox").value()
        min_val = getattr(self.ui, f"{name}_min_spinbox").value()
        spinbox = getattr(self.ui, f"{name}_spinbox")
        slider = getattr(self.ui, f"{name}_slider")
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
        if self.ui.filters_list.currentRow() < 0:
            logger.info("Filter list empty")
            return
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
            elif algo_type == "remez":
                filter = RemezFilter.create(filter.name)
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
        filter.decimation = self.ui.decimation_spinbox.value()
        fs = self.ui.sampling_freq_spinbox.value() * 10 ** (3 * self.ui.sampling_freq_combobox.currentIndex())
        # item.setText(f"{str(filter)}, {filter.n_taps * fs * 1e-9:.2f} GMAC/s")
        item.setData(QtCore.Qt.UserRole, filter)
        if update_widgets:
            self.select_filter()

        self.start_design()
        self.calc_compute()


    def start_design(self):
        if self.ui.filters_list.currentRow() < 0:
            logger.info("Filter list empty")
            return
        filt = self.ui.filters_list.currentItem().data(QtCore.Qt.UserRole)
        coe = filt.generate()
        coe_str = ", ".join([f"{c:.3e}" for c in coe])
        logger.info(f"Generating {filt.name} filter: {filt.n_taps} taps using {filt.algo}.")
        self.ui.coe_label.setText(coe_str)
        self.ui.coefficients_name_label.setText(f"{filt.name} coefficients:")
        with block_signals(self.ui.ntaps_spinbox):
            self.ui.ntaps_spinbox.setValue(filt.n_taps)
        max_val = self.ui.ntaps_max_spinbox.value()
        min_val = self.ui.ntaps_min_spinbox.value()
        cal_val = filt.n_taps
        slider_val = int(100 * (cal_val - min_val) / (max_val - min_val))
        with block_signals(self.ui.ntaps_slider):
            self.ui.ntaps_slider.setValue(slider_val)

        self.update_plot()
        self.model.update_filtered_amplitudes()
        return

    def calc_compute(self):
        fs = self.ui.sampling_freq_spinbox.value() * 10 ** (3 * self.ui.sampling_freq_combobox.currentIndex())
        comp_tot = 0
        for fi in range(self.ui.filters_list.count()):
            item: QtWidgets.QListWidgetItem = self.ui.filters_list.item(fi)
            filt_: FIRFilter = item.data(QtCore.Qt.UserRole)
            comp = filt_.n_taps * fs
            comp_tot += comp
            fs /= filt_.decimation
            item.setText(f"{str(filt_)}, {comp * 1e-9:.2f} GMAC/s")
        self.ui.total_compute_label.setText(f"{comp_tot * 1e-9:.2f} GMAC/s")


    def add_filter(self, filt: FIRFilter=None):
        logger.info(f"Adding filter {self.filter_counter}")
        if not isinstance(filt, FIRFilter):
            if self.ui.filters_list.currentRow() >= 0:
                filt = copy.deepcopy(self.ui.filters_list.currentItem().data(QtCore.Qt.UserRole))
                filt.name = f"Filter {self.filter_counter}"
            else:
                logger.info("Filter list empty")
                filt = FirwinFilter(f"Filter {self.filter_counter}")
        item = QtWidgets.QListWidgetItem(f"{str(filt)}")
        item.setData(QtCore.Qt.UserRole, filt)
        with block_signals(self.ui.filters_list):
            self.ui.filters_list.addItem(item)
            self.ui.filters_list.setCurrentItem(item)
        # Add filter to plots
        plot = self.ui.plot0_widget.plot(antialias=True, name=filt.name)
        plot.setLogMode(False, False)
        color_ind = self.filter_counter % len(self.color_dict)
        color = list(self.color_dict.values())[color_ind]
        plot.setPen(pq.mkPen(color=color, width=2.0))
        self.amp_plot_list.append(plot)

        plot = self.ui.plot2_widget.plot(antialias=True, name=filt.name)
        plot.setLogMode(False, False)
        color_ind = self.filter_counter % len(self.color_dict)
        color = list(self.color_dict.values())[color_ind]
        plot.setPen(pq.mkPen(color=color, width=2.0))
        self.ph_plot_list.append(plot)

        plot = self.ui.plot1_widget.plot(antialias=True, name=filt.name)
        plot.setLogMode(False, False)
        color_ind = self.filter_counter % len(self.color_dict)
        color = list(self.color_dict.values())[color_ind]
        plot.setPen(pq.mkPen(color=color, width=2.0))
        self.dec_plot_list.append(plot)
        self.filter_counter += 1
        logger.info(f"Filter #{self.filter_counter} / {len(self.amp_plot_list)}")
        self.select_filter()
        self.update_parameters()

    def remove_filter(self):
        row = self.ui.filters_list.currentRow()
        if row >= 0:
            self.ui.filters_list.takeItem(row)
            plt = self.amp_plot_list.pop(row)
            self.ui.plot0_widget.removeItem(plt)
            plt = self.dec_plot_list.pop(row)
            self.ui.plot1_widget.removeItem(plt)

    def update_plot(self):
        item = self.ui.filters_list.currentItem()
        ind = self.ui.filters_list.currentRow()
        filt: FIRFilter = item.data(QtCore.Qt.UserRole)
        w, h = ss.freqz(filt.coe, worN=self.ui.nfreq_spinbox.value())
        f = w / np.pi
        logger.debug(f"Updating plot. Filter #{ind} / {len(self.amp_plot_list)}: {filt.name}")
        self.amp_plot_list[ind].setData(f, 20*np.log10(np.abs(h)))
        self.ph_plot_list[ind].setData(f, (np.angle(h)) * 180 / np.pi)
        self.update_dec_plot()

    def update_dec_plot(self):
        fs = self.ui.sampling_freq_spinbox.value() * 10 ** (3 * self.ui.sampling_freq_combobox.currentIndex())
        freq = fs * np.linspace(0, 1, self.ui.nfreq_spinbox.value()) / 2
        fa = freq
        h_tot = np.zeros_like(freq)
        h_list = [h_tot]
        for fi in range(self.ui.filters_list.count()):
            k = np.round(freq / fs)
            fa = np.abs(freq  - k * fs)
            filt: FIRFilter = self.ui.filters_list.item(fi).data(QtCore.Qt.UserRole)
            w, h = ss.freqz(filt.coe, worN=self.ui.nfreq_spinbox.value())
            f = w / np.pi
            h_amp = 20 * np.log10(np.abs(h))
            filt_interp = interp1d(f, h_amp, bounds_error=False, fill_value="extrapolate")
            h_int = filt_interp(2 * fa / fs)
            h_tot += h_int
            h_list.append(np.copy(h_tot))
            self.dec_plot_list[fi].setData(freq, h_list[-1])
            fs /= filt.decimation
        self.update_freq_plot()
        return fa

    def update_freq_plot(self):
        points = []
        n_row = self.model.rowCount()
        logger.info(f"Plotting {n_row} frequencies")
        for row in range(n_row):
            x = self.model.data(self.model.index(row, self.model.COL_FREQ), QtCore.Qt.UserRole)
            y = self.model.data(self.model.index(row, self.model.COL_AMP), QtCore.Qt.UserRole)
            d = {"pos": (x,y),
                 "size": 10,
                 "pen": pyqtgraph.mkPen(self.highlight_colors[row % len(self.highlight_colors)], width=2),
                 "brush": None
                 }
            points.append(d)
            x = self.model.data(self.model.index(row, self.model.COL_ALIAS), QtCore.Qt.UserRole)
            y = self.model.data(self.model.index(row, self.model.COL_FILTERED), QtCore.Qt.UserRole)
            d = {"pos": (x,y),
                 "size": 10,
                 "pen": None,
                 "brush": pyqtgraph.mkBrush(self.highlight_colors[row % len(self.highlight_colors)])
                 }
            points.append(d)
        self.freq_scatterplot.setData(points)

    def select_filter(self):
        if self.ui.filters_list.currentRow() < 0:
            logger.info("Filter list empty")
            return
        item = self.ui.filters_list.currentItem()
        row = self.ui.filters_list.currentRow()
        filt: FIRFilter = item.data(QtCore.Qt.UserRole)
        logger.info(f"select_filter {item}: {filt}, {row}, {self.ui.filters_list.item(row)}")
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
        with block_signals(self.ui.decimation_spinbox):
            self.ui.decimation_spinbox.setValue(filt.decimation)
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

        coe_str = ", ".join([f"{c:.3e}" for c in filt.coe])
        self.ui.coe_label.setText(coe_str)
        self.ui.coefficients_name_label.setText(f"{filt.name} coefficients:")

    def set_freq_suffix(self, value=None):
        if not isinstance(value, float):
            value = 10 ** (3 * self.ui.sampling_freq_combobox.currentIndex())
        logger.info(f"New suffix for {value}: {self.ui.sampling_freq_combobox.currentText()}")
        old_value = self.model.suffix_factor
        self.model.update_suffix(value)
        self.ui.sampling_freq_spinbox.setValue(self.ui.sampling_freq_spinbox.value() / old_value / value)

    def set_sampling_freq(self, value=None):
        if not isinstance(value, float):
            value = self.ui.sampling_freq_spinbox.value()
        logger.info(f"New sampling frequency: {value} {self.ui.sampling_freq_combobox.currentText()}")
        self.model.update_alias_freq()
        self.update_dec_plot()

    def add_freq(self):
        f_name = f"f{self.freq_counter}"
        logger.info(f"Adding frequency {f_name}")
        color = self.highlight_colors[self.freq_counter % len(self.highlight_colors)]
        self.model.addRow(f_name, 100 * 10 ** (3 * self.ui.sampling_freq_combobox.currentIndex()), 0, color)
        self.freq_counter += 1
        self.update_freq_plot()

    def remove_freq(self):
        ind = self.ui.freq_table.currentIndex()
        if ind.isValid():
            self.model.removeRow(ind.row())
        self.update_freq_plot()

    def calc_freq_response(self, freq, amp):
        fs = self.ui.sampling_freq_spinbox.value() * 10 ** (3 * self.ui.sampling_freq_combobox.currentIndex())
        logger.debug(f"resp for {freq}")
        h = 1
        for fi in range(self.ui.filters_list.count()):
            k = np.round(freq / fs)
            freq = np.abs(freq  - k * fs)
            w = freq * 2 * np.pi / (fs)
            filt: FIRFilter = self.ui.filters_list.item(fi).data(QtCore.Qt.UserRole)
            hf = np.sum(filt.coe * np.exp(-1j * w * np.arange(len(filt.coe))))
            h *= hf
            logger.debug(f"Filter {filt.name}: w {w:.2f} hf {20 * np.log10(np.abs(hf)):.2f} dB")
            fs /= filt.decimation
        return amp + 20 * np.log10(np.abs(h))

    def calc_alias_freq(self, freq):
        fs = self.ui.sampling_freq_spinbox.value() * 10 ** (3 * self.ui.sampling_freq_combobox.currentIndex())
        fa = freq
        for fi in range(self.ui.filters_list.count()):
            k = np.round(fa / fs)
            fa = np.abs(fa  - k * fs)
            filt: FIRFilter = self.ui.filters_list.item(fi).data(QtCore.Qt.UserRole)
            fs /= filt.decimation
        return fa

    def save_filters(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(parent=self, caption="Save filter set as yaml",
                                                            directory=str(self.last_load_dir),
                                                            filter="yaml files (*.yml);;All files (*)")
        filename = pathlib.Path(filename)
        self.last_load_dir = filename.parent
        logger.info(f"Saving filters to {filename}")
        d = dict()
        for fi in range(self.ui.filters_list.count()):
            filt: FIRFilter = self.ui.filters_list.item(fi).data(QtCore.Qt.UserRole)
            d[filt.name] = filt.to_dict()
        s = yaml.safe_dump(d, sort_keys=False)
        with open(filename, "w", encoding="utf-8") as fd:
            fd.write(s)

    def load_filters(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(parent=self, caption="Save filter set as yaml",
                                                            directory=str(self.last_load_dir),
                                                            filter="yaml files (*.yml);;All files (*)")
        filename = pathlib.Path(filename)
        self.last_load_dir = filename.parent
        logger.info(f"Loading filters from {filename}")
        with open(filename, "r") as fd:
            s = fd.read()
        d = yaml.safe_load(s)
        for fk in d.keys():
            filt = FIRFilter.from_yaml(d[fk])
            self.add_filter(filt)

    def save_freqs(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(parent=self, caption="Save frequency set as yaml",
                                                            directory=str(self.last_load_dir),
                                                            filter="yaml files (*.yml);;All files (*)")
        filename = pathlib.Path(filename)
        self.last_load_dir = filename.parent
        logger.info(f"Saving frequencies to {filename}")
        d = dict()
        for row in range(self.model.rowCount()):
            name = self.model.data(self.model.index(row, self.model.COL_NAME), QtCore.Qt.UserRole)
            freq = self.model.data(self.model.index(row, self.model.COL_FREQ), QtCore.Qt.UserRole)
            amp = self.model.data(self.model.index(row, self.model.COL_AMP), QtCore.Qt.UserRole)
            unit = self.ui.sampling_freq_combobox.currentText()
            suffix_factor = self.model.suffix_factor
            d[name] = {"freq": freq * suffix_factor, "amp": amp, "unit": unit}
        s = yaml.safe_dump(d, sort_keys=False)
        with open(filename, "w", encoding="utf-8") as fd:
            fd.write(s)

    def load_freqs(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(parent=self, caption="Save frequency set as yaml",
                                                            directory=str(self.last_load_dir),
                                                            filter="yaml files (*.yml);;All files (*)")
        filename = pathlib.Path(filename)
        self.last_load_dir = filename.parent
        logger.info(f"Loading frequencies from {filename}")
        with open(filename, "r") as fd:
            s = fd.read()
        d = yaml.safe_load(s)
        for name in d.keys():
            f = d[name]["freq"]
            a = d[name]["amp"]
            u = d[name]["unit"]
            ind = self.model.suffix_list.index(u)
            suffix_factor = 10 ** (3 * ind)
            color = self.highlight_colors[self.freq_counter % len(self.highlight_colors)]
            self.model.addRow(name, f * suffix_factor, a, color)
            self.freq_counter += 1
        self.update_freq_plot()

    def closeEvent(self, a0, QCloseEvent=None):
        logger.info(f"Saving settings.")
        self.settings.setValue("ntaps", self.ui.ntaps_spinbox.value())
        self.settings.setValue("nfreq", self.ui.nfreq_spinbox.value())
        self.settings.setValue("sampling_freq", self.ui.sampling_freq_spinbox.value())
        self.settings.setValue("sampling_freq_suffix", self.ui.sampling_freq_combobox.currentText())
        self.settings.setValue("splitter", self.ui.splitter.saveState())
        self.settings.setValue("splitter2", self.ui.splitter_2.saveState())
        self.settings.setValue("load_dir", self.last_load_dir)


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