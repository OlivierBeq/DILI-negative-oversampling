# Custom oversampling technique for negative samples.


This repository contains all the code that was used to obtain the results presented in [Keep looking at the negative side](). 

## Getting started

### Setting up an environment

One can install all the required dependencies as follows:
```
git clone https://github.com/OlivierBeq/DILI-negative-oversampling
cd DILI-negative-oversampling
conda create -n DILInegOS python=3.11 -q -y
pip install -r requirements.txt
conda install notebook ipykernel ipywidgets
```

### Download data

The data **required** to run the code contained in this repository is available on [Zenodo](https://zenodo.org/records/15091446).
Two files can be downloaded:
- `data.zip` (this contains the 3 files that all the analysis relies on),
  - `Supp. File 1.xlsx`
  - `DILI_dataset_ECFP6_FCFP6.tsv`
  - `Papyrus_05-7_BioSpectra_protein-descriptors-ZscalesVanWesten.tsv.gz`
  - ... and other files derived from these three (the notebooks can recreate them) that results are obtained from.
- `results.zip` (this corresponds to the `results` folder created when running notebooks).

Once downloaded, place the unzipped `data` folder in the `src/DILI-negative-oversampling` folder.

Placing the unzipped `results` folder in the `src/DILI-negative-oversampling` folder, will result in the same as running all notebooks.

### Running notebooks

1. Run the Python notebooks in order **from 1 to 7**.
<br/>:warning: Running notebook **4** is very computationally heavy (~200GB RAM & > 10 cores).
<br/>This notebook **4** can be skipped.
2. Figures and Tables will be either presented in the notebooks or saved in the folder `src/DILI-negative-oversampling/results`.


**By default, files requiring substantial resources to be obtained are used as is.
To force them to be re-calculated, run notebook `XX-Advanced_removal_to_run_from_scratch`, before running notebooks 1 to 7 (notebook 4 cannot be skipped in this case) .**
