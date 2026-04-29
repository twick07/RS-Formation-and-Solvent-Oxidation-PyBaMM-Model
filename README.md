# RS Formation and Solvent Oxidation PyBaMM Model

Code associated with the manuscript:

**'A Continuum Scale Model for Rocksalt Formation and Solvent Oxidation in Ni-Rich Lithium-Ion Batteries'**

## Overview

This repository contains the scripts used to generate all results figures in the main manuscript and supplementary information. The model is implemented using a modified version of [PyBaMM](https://github.com/twick07/pybamm_wickramanayake2026) (Python Battery Mathematical Modelling).

## Scripts

| Script | Figures |
|--------|---------|
| `Fig_1_2_Half_Cell_Model_Operation_Wickramanayake_2026_2.py` | Figures 1–2 |
| `Fig_3_Script_Half_Cell_Mechanistic_Validation_Wickramanayake_2.py` | Figure 3 |
| `Fig_4_5_6_Script_Half_Cell_Cycling_Results_Wickramanayake_2.py` | Figures 4–6 |
| `Fig_7_8_9_Script_Full_Cell_Cycling_Results_Wickramanayake_2.py` | Figures 7–9 |
| `Fig_10_Internal_Pressure_Change_Plot_Wickramanayake_2.py` | Figure 10 |
| `SI_Original_SC_Model_Validation_Wickramanayake2026.py` | Supplementary Information |
| `Parameter_Sets_Wickramanayake2026.py` | Parameter sets (imported by all figure scripts) |

Experimental and parameter data required by the scripts are located in the `Data/` folder.

## Installation

### Prerequisites

Before starting, ensure the following are installed on your system:

- **Python 3.12** — download from [python.org](https://www.python.org/downloads/)
- **Git** — download from [git-scm.com](https://git-scm.com/downloads). Note: GitHub Desktop does not add Git to the system PATH, so a standalone Git installation is required.

### 1. Clone this repository

```bash
git clone https://github.com/twick07/RS-Formation-and-Solvent-Oxidation-PyBaMM-Model.git
cd RS-Formation-and-Solvent-Oxidation-PyBaMM-Model
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including the modified PyBaMM version used in this work.

### 4. Run a script

```bash
python Fig_1_2_Half_Cell_Model_Operation_Wickramanayake_2026_2.py
```
