## **DDVAR:** **A Difference-Aware Framework for Disease Druggable Variant Prioritization**

## 🚀 Overview

<img src="assets/image-20260423205201616.png" alt="image-20260423205201616" style="zoom: 80%;" />

**DDVAR** is a multimodal difference-aware framework for prioritizing potentially druggable variants.  It integrates **sequence**, **secondary structure (DSSP)**, **3D structural**, and **evolutionary (PSSM)** features within a unified **sequence–graph architecture**, while explicitly modeling the differences between **wild-type** and **mutant** proteins.

Main experimental results as follows :

- **Structural features** provide the most discriminative signal

-  DDVAR captures both **local mutation effects** and **long-range structural changes**

-  The framework can be effectively adapted to **disease-specific settings** with limited data

-  DDVAR also supports **large-scale variant prioritization on gnomAD**, enabling downstream analysis at the **gene**, **protein**, and **mutation** levels

  

## 📂 Project Structure

```
DDVAR/
│
├── MainExperiments/                        # Main experiment
│   ├── code/
│   │   ├── model/                          # Model definitions
│   │   │   └── model_diff.py
│   │   └── train.py                        # Training and reproduction script
│   ├── out/                                # Experiment outputs
│   ├── log/                                # Runtime logs
│   ├── save_model/                         # Directory for saved models
│   │   └── run_20260330-1020/              # Model checkpoint for reproduction
│   │
│   └── tmp/                                # Data required for reproduction
│       ├── diff_negative_alt_graphs_mul_1_thre_10.pt
│       ├── diff_negative_ref_graphs_mul_1_thre_10.pt
│       ├── diff_positive_alt_graphs_thre_10.pt
│       └── diff_positive_ref_graphs_thre_10.pt
│
└── ObtainFeature/                          # Feature extraction
    ├── code/
    │   ├── hub/                           
    │   │   ├── checkpoints/                # Pretrained model checkpoints
    │   │   └── facebookresearch_esm_main/  # ESM protein language model
    │   └── obtain_feature_pipeline.py      # Main feature extraction pipeline
    └── out/                                # Feature extraction outputs
        ├── contact/                        # Residue contact graphs
        ├── csv/                            # Mutation data in CSV format
        ├── dssp/                           # DSSP secondary structure
        ├── pdb/                            # PDB structures
        ├── prt_asseq/                      # Protein sequences
        ├── prt_repr/                       # Protein representations
        └── pssm/                           # Position-specific scoring matrices
```





## 📌 Reproducing Paper Results

### ⚙️Requirements

Install the required dependencies:

```
numpy==1.22.4
pandas==2.0.3
scipy==1.11.1
scikit-learn==1.3.0
torch==1.13.1
torch-geometric==2.6.0
biopython==1.81
tqdm==4.65.0
```

Reproducing the results reported in the paper **does NOT require feature extraction**.  We have already provided:

- Preprocessed input features
- The best trained model checkpoint

📁 Paths:

- Input features:   `MainExperiments/tmp/`
- Pretrained model:   `MainExperiments/save_model/run_20260330-1020/`

▶️ Run reproduction

```
cd MainExperiments/code
python train.py --mode=reproduce
```

------

## 📌 Training on Custom Data

If you want to train DDVAR on your own dataset, follow the steps below.

### 📦 Step 1: Data Preparation

Prepare your raw mutation file `Testset` and perform **ANNOVAR annotation**. The following files are required:

- `Testset_asseq`
- `Testset.exonic_variant_function`

Place them under `DDVAR/ObtainFeature/out/prt_asseq/Testset/`

------

### ⚙️ Step 2: Feature Extraction

#### 2.1 Sequence Extraction

Extract mutation-centered protein sequences:

```
cd ObtainFeature/code
python obtain_feature_pipeline.py --step=obtain_prt_cut
```

------

#### 2.2 Multimodal Feature Extraction

Before running this step, ensure the following dependencies are prepared:

- PSI-BLAST database
- DSSP executable (`mkdssp`)
- ESMFold pretrained weights
- ESM2 pretrained weights

ESMFold and ESM2 weights should be manually downloaded and placed under `ObtainFeature/code/hub/`.

```
python obtain_feature_pipeline.py \
    --mkdssp_path /path/mkdssp \
    --psiblast_db_path /path/blast/db/swissprot \
    --step=obtain_mutilmodal_features
```

------

### 🧠 Step 3: Model Training

After feature extraction, train the model using:

```
cd MainExperiments/code
python train.py --mode=train
```

📁 Output:

- Logs:  `MainExperiments/out/log/`
- Saved models:   `MainExperiments/out/save_model/`