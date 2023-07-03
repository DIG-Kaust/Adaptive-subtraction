![LOGO](https://github.com/DIG-Kaust/Adaptive-subtraction/blob/master/figures/logo.png)

Reproducible material to perform adaptive subtraction to correct predicted multiples. Synthetic seismic data is modeled and multiples are predicted following the SRME method. We tried using an L1-L1 approach, where the optimization on the patched data is conducted by the ADMM. Additionally, the Curvelet Transform is used for demultiple.

## Project structure
This repository is organized as follows:

* :open_file_folder: **adasubtraction**: python library containing routines to step-by-step process to remove multiples from synthetic and real data;
* :open_file_folder: **data**: folder containing the SEAM Phase 2D Velocity Model. You can find instructions to directly download the data;
* :open_file_folder: **figures**: folder containing some figures used in the notebooks;
* :open_file_folder: **notebooks**: set of jupyter notebooks compressing data generation, surface related multiples prediction and multiples elimination;
* :open_file_folder: **scripts**: set of python scripts used to to forward seismic modeling and to predict multiples; 

## Notebooks
The following notebooks are provided:

- :orange_book: ``Data_Modeling.ipynb``: notebook doing modeling of seismic data and primaries with ghosts with the Devito engine;
- :orange_book: ``Multiples_Prediction.ipynb``: notebook carrying out multidimensional convolution with a full seismic data set to predict surface related multiples;
- :orange_book: ``Adaptive_Subtraction.ipynb``: notebook performing adaptive subtraction on with the ADMM in a 1D example (as in Guitton and Verschuur, 2004), on synthetic seismic CSGs and on the Voring dataset;
- :orange_book: ``Primary_Multiple_Curvelet_Separation.ipynb``: notebook generating masks with predicted multiples and conducting separation of primaries and multiples in the curvelet domain;

## Scripts
The following scripts are provided:

- :orange_book: ``model_data.py``: wrapper for seismic modeling of a full 2D dataset. Recommended to run this script on the terminal instead of running the code on the notebooks.
- :orange_book: ``create_multiples.py``: wrapper for SRME prediction using MDC. Recommended to run this script on the terminal instead of running the code on the notebooks.

## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment. Note that for the notebook ``Primary_Multiple_Curvelet_Separation.ipynb`` is neccesary to use a different environment (instructions are provided).

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

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX 3090 GPU. Different environment configurations may be required for different combinations of workstation and GPU.
