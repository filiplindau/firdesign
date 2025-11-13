"""
Calculate FIR filters

:created: 2025-11-12
:author: Filip Lindau
"""

import scipy.signal as ss
from enum import Enum
import logging

log_format = "%(asctime)s - %(levelname)s - %(name)s %(funcName)s - %(message)s"

# Set up the logging configuration using basicConfig
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FilterType(Enum):
    LOWPASS = 0
    HIGHPASS = 1
    BANDPASS = 2


class FIRFilter:
    def __init__(self):
        self.coefficients = None

    def calc_coefficients(self, n, f0, f1, filter_type=FilterType.LOWPASS):
        self.coefficients = ss.firwin(n, f0, f1)
