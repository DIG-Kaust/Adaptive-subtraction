![LOGO](https://github.com/DIG-Kaust/Adaptive-subtraction/blob/master/figures/logo.png)

Reproducible material to perform adaptive subtraction to correct predicted multiples. We tried using an L1-L1 (data misfit and regularization) approach, where the optimization on the patched data is conducted by the ADMM and also compared with the traditional LSQR.

## Project structure
This repository is organized as follows:

* :open_file_folder: **adasubtraction**: python library containing routines to apply adaptive subtraction on shot gathers;
* :open_file_folder: **data**: folder containing the Voring dataset and testing arrays in npz files;
* :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments for different sizes of data;
* :open_file_folder: **scripts**: set of python scripts used to create multiples prediction and run the multiple experiments with different input parameters for adaptive subtraction studies; 
* :open_file_folder: **figures**: folder where figures from varios script experiments will be saved;

## Notebooks
The following notebooks are provided:

- :orange_book: ``Adaptive_subtraction_1d.ipynb``: notebook explaining the ADMM theory and reproducing an 1D experiment 
as in Guitton et al., 2004;
- :orange_book: ``Adaptive_subtraction_patch.ipynb``: notebook computing matching filters in a small patch of a shot gather of the
Voring data and comparing results of different algorithms;
- :orange_book: ``Adaptive_subtraction.ipynb``: notebook applying adaptive subtraction on a complete shot gather by patching data and doing qc on common channel gathers obtained with the lsqr on all shots;
- :orange_book: ``Adaptive_subtraction_synthetic_shot.ipynb``: notebook performing ADMM on a synthetic 2d dataset through simple tests to proof capacities of the L1-L1 optimization;
- :orange_book: ``SRME.ipynb``: notebook where an initial multiple prediction is produced with the SRME method. Different approches of the ADMM applied to the data are introduced;

## Scripts
The following scripts are provided:

- :orange_book: ``push_rho.py``: fix epsilon (reg. parameter) and change rho values to observe how fast is the convergence of the solution. The results are stored in an npz file.
- :orange_book: ``tests.py``: change input parameters of the ADMM to correct input multiples and store differences with true multiples in a dictionary.
- :orange_book: ``create_multiples.py``: create initial prediction of multiples performing a Multidimensional Convolution.

## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. After that you can simply install your package:
```
pip install adasubtraction
```
or in developer mode:
```
pip install -e adasubtraction
```

Remember to always activate the environment by typing:
```
conda activate adasubtraction
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX 3090 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.
