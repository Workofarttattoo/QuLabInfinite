
# Frequency Lab

The Frequency Lab is a software toolkit for interacting with Software-Defined Radios (SDRs) for the purpose of capturing, analyzing, and transmitting radio frequency (RF) signals. It is designed to be a modular and extensible platform for radio experimentation and research.

## Features

- **SDR Agnostic Interface**: A simple interface for connecting to various SDRs, with initial support for a "dummy" driver for testing without hardware.
- **Signal Analysis**: Core signal processing functions, including Power Spectral Density (PSD) calculation using Welch's method.
- **Command-Line Interface**: A CLI tool for basic operations like scanning a frequency and visualizing the spectrum.
- **Extensible Design**: The lab is structured to be easily extended with new signal processing modules, SDR drivers, and analysis routines.

## Getting Started

### Prerequisites

- Python 3.6+
- NumPy
- SciPy
- Matplotlib

To install the required libraries:
```bash
pip install numpy scipy matplotlib
```

### Running the Lab

The primary way to interact with the lab is through the command-line interface.

**To scan a frequency:**

```bash
python -m frequency_lab.cli --freq <frequency_in_hz>
```

For example, to scan an FM radio station at 101.1 MHz:

```bash
python -m frequency_lab.cli --freq 101.1e6
```

This will open a plot showing the spectrum of the captured signal.

### CLI Options

- `--mode`: `scan` or `tx`. `scan` is for receiving and analyzing. `tx` is for transmitting (use with caution).
- `--freq`: Center frequency in Hz.
- `--bw`: Bandwidth in Hz.
- `--sr`: Sample Rate in Hz.
- `--driver`: The SDR driver to use (e.g., `uhd`, `soapy`, or `dummy`).
- `--serial`: The serial number of the SDR device.


## Running Tests

To run the included unit tests, navigate to the root directory of the project and run:

```bash
python -m unittest discover -s frequency_lab
```

## Disclaimer

**WARNING**: Transmitting radio signals without the proper licensing and knowledge is illegal in most countries and can cause harmful interference to critical communication systems. The transmission capabilities of this software are for educational and research purposes only and should be used with extreme caution. Always operate within the legal limits of your jurisdiction and use appropriate hardware (like a dummy load or attenuator) when testing.
