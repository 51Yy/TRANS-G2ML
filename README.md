## Machine Learning-Guided Design of Low Immunogenic Polymers to Enhance T Cell Responses in mRNA Cancer Vaccination

<img width="451" height="547" alt="image" src="https://github.com/user-attachments/assets/43761530-8d23-4b35-b1f4-6dc793665e71" />

mRNA therapeutics require a careful balance between antigen expression and immune activation. Excessive immunogenicity can induce T-cell exhaustion and reduce mRNA stability, while insufficient activation may lead to immune tolerance.

This repository provides code and data for an accelerated discovery platform that integrates:

- Combinatorial polymer chemistry
- High-throughput experimental screening
- Descriptor-based Generation-1 ML (DESC-G1ML)
- Transformer-based Generation-2 ML (TRANS-G2ML)

The platform is designed to identify and optimize polymer carriers that promote productive and sustained T-cell activation by tuning mRNA antigen expression and immunostimulatory response.

## Model Description

### DESC-G1ML

A descriptor-based machine learning model trained on high-throughput experimental data to predict polymer delivery performance.

### TRANS-G2ML

TRANS-G2ML is a customized transformer-based model developed based on the TransPolymer framework. The initial running environment and dependency setup were established following the original TransPolymer repository:

> https://github.com/ChangwenXu98/TransPolymer

## Usage

### DESC-G1ML

Open the following notebook in Jupyter Notebook or JupyterLab:

```text
scripts/DESC_G1ML_training.ipynb
```

### TRANS-G2ML

Run downstream single-task learning:

```bash
python ./scripts/TRANS-G2/Downstream_STL.py
```

Run downstream multi-task learning:

```bash
python ./scripts/TRANS-G2/Downstream_MTL.py
```

## Data

Experimental records are provided in:

```text
data/Data_records.xlsx
```

Additional datasets for TRANS-G2ML are located in:

```text
scripts/TRANS-G2/data/
```

## Acknowledgements

The TRANS-G2ML model in this repository is customized based on the TransPolymer model.

Original paper:

> Xu, C. et al. *A transferable, data-efficient and scalable deep learning framework for predicting polymer properties.*  
> **npj Computational Materials** 9, 64, 2023.  
> https://www.nature.com/articles/s41524-023-01016-5

Original repository:

> https://github.com/ChangwenXu98/TransPolymer

## Citation

If you use this repository, please cite the original TransPolymer work and this work once it becomes available.

### Original TransPolymer work

```bibtex
@article{xu2023transpolymer,
  title={TransPolymer: a Transformer-based language model for polymer property predictions},
  author={Xu, Changwen and Wang, Yuyang and Barati Farimani, Amir},
  journal={npj Computational Materials},
  volume={9},
  number={1},
  pages={64},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

### This work

This work is currently under review. Citation information will be updated upon publication.

```text
Machine Learning-Guided Design of Low Immunogenic Polymers to Enhance T Cell Responses in mRNA Cancer Vaccination.
Under review.
```
