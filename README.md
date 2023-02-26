![LOGO](https://github.com/DIG-Kaust/Adaptive-subtraction/blob/main/logo.png)

Reproducible material to perform adaptive subtraction on seismic data given initial estimate of multiples. It's objective is to use the ADMM 
algorithm to solve the optimization problem of the matching-filter approach posed as an L1-L1 regression problem.

## Project structure
This repository is organized as follows:

* :open_file_folder: **adasubtraction**: python library containing routines for adaptive subtraction on shot gathers;
* :open_file_folder: **data**: folder containing the Voring dataset and testing arrays in npz files;
* :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments for different sizes of data;
* :open_file_folder: **scripts**: set of python scripts used to run the multiple experiments with different input parameters for adaptive subtraction studies; 
* :open_file_folder: **figures**: folder where figures from varios script experiments will be saved;

## Notebooks
The following notebooks are provided:

- :orange_book: ``Adaptive_subtraction_1d.ipynb``: notebook explaining the ADMM theory and reproducing an 1D experiment 
as in Guitton et al., 2004;
- :orange_book: ``Adaptive_subtraction_patch.ipynb``: notebook computing matching filters in a small patch of a shot gather of the
Voring data and comparing results of different algorithms;
- :orange_book: ``Adaptive_subtraction.ipynb``: notebook applying adaptive subtraction on a complete shot gather by patching data and doing
qc on common channel gathers obtained with the lsqr on all shots;
- :orange_book: ``Adaptive_subtraction_tests.ipynb``: notebook performing several tests varying input parameters of the ADMM applied 
over all shot gathers and then plotting common channel gathers qc results;


## Scripts
The following scripts are provided:

- :orange_book: ``read_gathers.py``: script reading input shot gathers and storing them in a 3d array;
- :orange_book: ``adaptive_subtraction_3d.py``: script performing adaptive subtraction on a cube of shot gathers.
- :orange_book: ``adaptive_subtraction_qc.py``: script doing qc on common channel gathers by plotting correlation of traces and estimating
amplitude average of multiples present in arrays of total data and primaries data.

The last script require an input ``lsqr.npz`` file containing the following fields. 

- :card_index: ``multiples_lsqr``: 3-dimensional corrected multiples data using lsqr of size ``ns x nr x nt``.
- :card_index: ``primaries_lsqr``: 3-dimensional corrected primaries data using lsqr of size ``ns x nr x nt``.


## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```

Remember to always activate the environment by typing:
```
conda activate adasubtraction
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX 3090 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.
