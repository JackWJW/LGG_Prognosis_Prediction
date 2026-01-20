# LGG_Prognosis_Prediction

## Abstract
Glycosphingolipids (GSLs) are essential elements of biological membranes, with import roles in cancer. Disrupted GSL metabolism is associated with malignancy and severity across a range of different cancers, with different GSLs implicated in different tumours. GSLs have potential mechanistic roles in Glioblastoma Multiforme (GBM), however their functions in Lower Grade Gliomas (LGGs) are less well studied. Here we present ensemble machine learning approaches built on transcriptomic data for LGG alongside GSL-specific metabolic simulations to predict survival outcomes in LGG. The ensemble approach demonstrates effective risk stratification for LGG patients. We developed a python package to facilitate easy implementation of GSL-specific metabolic modelling strategies and to deploy our models to make risk predictions based on RNA-seq data. The resulting model derived risk groups offered biological insights to the potential roles of GSLs in LGG, highlighting cell motility, cell division, Wnt signalling and microtubule organisation as areas of interest for further research. We propose that GSL-based diagnostics and/or prognostics may prove clinically beneficial given the well-established shedding of GSLs into the tumour microenvironment as a route for detection paired with high performing machine learning approaches to model patient outcomes.

## Project Schematic
![Project Schema](./Figures/Workflow_Schematic.png)

## Repo Outline
- Differential Expression Analysis scripts and results for LGG vs GBM are found in:
    - >Glioma_DEA

- The survival analysis script is found in:
    - >Survival_Analysis

- TCGA input datasets are found in:
    - >TCGA_Data

- CGGA input datasets are found in:
    - >CGGA_Data

- TCGA data preparation simulation and integration files are found in: 
    - >TCGA_Simulation_Integration

- Flux simulation integrated datasets can be found in:
    - >iMAT_integrated_data

- CGGA data preparation simulation and integration files are found in 
    - >CGGA_Simulation_Integration

- The scripts for performing the nested cross-validated training and optimisation of the ensemble machine learning approach against only TCGA data to determine best simulation parameters are found in 
    - >TCGA_Nested_CV

- The results from the nested cross-validation experiment can be found in:
    - >LGG_CV_Prediction_Results

- The scripts for training the model on all TCGA data and subsequent validation against CGGA data can be found in:
    - >Full_TCGA-Train_CGGA-Validation

- The results from the CGGA validated experiment can be found in:
    - >CGGA_Validation_Results

- Analysis of the results can be found in:
    - >Results_Analysis
        - >DEA
        - >DAVID_Analysis
        - >ESTIMATE

- Preparation of models trained on both TCGA and CGGA data for deployment in pyGSLModel and upload to hugging face are found in:
    - >HF_Model_Prep
