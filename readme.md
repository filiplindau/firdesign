## Introduction

FIR filter designer using scipy.signal. Filters can be added and removed, with or without decimation.


## Usage

Filters are processed in sequence and progressively scaled in sample frequency according to decimation.
The generating algorithm can be selected together with various settings.

The central plots are for each filter individually. The right plot is for the combined filter effect.
Additionally, specific frequencies can be calculated, including the effect of aliasing, and are also plotted on the right plot.
Open circles show original frequency, filled circles show aliased frequency.
The attribute values (name, frequency, level) for these are edited in the table by double-clicking. 

![Sample design](./doc/lowpass_0.25.png?raw=true)

## Requirements

- PyQt5
- pyqtgraph
- scipy
- numpy