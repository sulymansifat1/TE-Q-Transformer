# Physics-Guided Temperature-Embedded Quantum Transformer for Lithium-Ion Battery State-of-Health Estimation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.35%2B-blueviolet.svg)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

TE-Q-Transformer is a hybrid quantum-classical deep learning framework for lithium-ion battery state-of-health (SOH) estimation. The codebase addresses cycle-level capacity degradation estimation by integrating physics-guided Arrhenius temperature transformations directly into a parameterized quantum circuit representation. The core implementation couples temperature-dependent degradation kinetics with a four-qubit simulated variational circuit, a 1D convolutional smoothing layer, and a multi-layer Transformer encoder backbone. This repository provides the complete model implementation, baseline architectures, data loading pipelines, evaluation routines, unit tests, and notebook workflows to reproduce the experimental pipeline.

---

## Repository Contents

- **Proposed Architecture (`models/`)**: Core implementation of the TE-Q-Transformer architecture, reference checkpoints, and configuration factories (`TEQTransformer`, `TEQTransformerConfig`, `rich_entangler_config`).
- **Workflow Notebooks (`notebooks/`)**: Self-contained Jupyter notebooks for full contemporary baseline benchmarking (10 baselines), ablation studies, generalization experiments, formal statistical significance validation, explainability analysis, and CALCE cross-cell evaluation.
- **Pre-Packaged Datasets (`datasets.zip`)**: Ready-to-upload archive containing all processed NASA/CALCE `.npy` arrays, baseline predictions, and model checkpoint for 1-click Kaggle reproduction.
- **Data Loaders & Preprocessing (`src/data/`)**: PyTorch Dataset classes, data loader generators, and zero-leakage feature scaling routines for NASA and CALCE battery datasets.
- **Evaluation Utilities (`src/eval/`)**: Standard regression metrics calculation (`RMSE`, `MAE`, `MAPE`, `R2`, `MaxE`).

---

## Model Architecture

![TE-Q-Transformer Architecture](assets/TE-Q-Architecture-Final.png)

---

## Repository Structure

```
.
├── README.md                                  # Repository documentation and Kaggle reproduction guide
├── datasets.zip                               # Self-contained archive for 1-click Kaggle Dataset upload (18.2 MB)
├── LICENSE                                    # MIT License
├── requirements.txt                           # Project dependencies
├── .gitignore                                 # Git ignore patterns
│
├── datasets/
│   ├── README.md                              # Dataset documentation and contracts
│   ├── NASA/
│   │   ├── processed/                         # Processed NASA arrays [N, 512, 4] and fitted scalers
│   │   └── predictions/                       # Baseline predictions CSV for statistical significance
│   └── CALCE/
│       ├── README.md                          # CALCE dataset provenance and details
│       └── processed/                         # Processed CALCE arrays [N, 512, 4] and metadata
│
├── models/
│   ├── __init__.py                            # Model package entry point
│   ├── checkpoints/
│   │   └── nasa_teq_transformer_best.pth      # Pretrained reference TE-Q-Transformer weights
│   └── proposed/
│       ├── __init__.py
│       └── te_q_transformer.py                # TE-Q-Transformer architecture and config
│
├── notebooks/
│   ├── benchmark/
│   │   └── baseline_comparison_nasa.ipynb     # NASA contemporary baseline comparison (10 baselines + TE-Q-Transformer)
│   ├── ablation/
│   │   └── ablation_study.ipynb               # Architectural component ablation experiments
│   ├── generalization/
│   │   └── generalization_study.ipynb         # Unseen-cell and temporal extrapolation experiments
│   ├── statistical_significance.ipynb         # Statistical hypothesis testing & block bootstrap (vs. iTransformer & QNN-GRU)
│   ├── explainability.ipynb                   # Integrated Gradients feature attributions & temperature sensitivity
│   └── Calce.ipynb                            # CALCE cross-cell and temporal extrapolation evaluation
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── nasa_loader.py                     # NASA dataset loaders and zero-leakage scaler
│   │   ├── calce_loader.py                    # CALCE dataset loaders
│   │   └── calce/
│   │       ├── __init__.py
│   │       └── preprocess.py                  # CALCE raw Arbin Excel extraction script
│   └── eval/
│       ├── __init__.py
│       └── metrics.py                         # RMSE, MAE, MAPE, R2, MaxE calculations
│
└── tests/
    ├── test_models.py                         # Forward pass verification for all architectures
    ├── test_datasets.py                       # Dataset file existence and tensor shape tests
    ├── test_scaling.py                        # Non-leakage feature scaling contract test
    └── test_calce_preprocess.py               # CALCE resampling and validation tests
```

---

## Running on Kaggle (Evaluation & Reproduction Guide)

To ensure effortless and 100% reproducible evaluation by reviewers without requiring local environment configuration, GPU driver setups, or quantum library compilation, **all experiments are designed to run directly inside Kaggle Notebooks**.

Every notebook is fully self-contained and equipped with **dynamic path resolution**—it automatically detects dataset arrays, model checkpoints, and source code directly from the uploaded dataset mount. Full evaluation metrics and publication-ready plots are printed directly in the notebook outputs, and all generated figures, tables, and reports are automatically packaged into a downloadable **`.zip` file** directly inside `/kaggle/working/`.

---

### Step 1: Upload `datasets.zip` as a Kaggle Dataset

All required processed data arrays, baseline prediction CSVs, the trained TE-Q-Transformer checkpoint, and source code are packaged into [`datasets.zip`](datasets.zip) (18.2 MB) located in the root of this repository.

1. Download or locate [`datasets.zip`](datasets.zip) from this repository.
2. In [Kaggle](https://www.kaggle.com/), click **+ Create** &rarr; **New Dataset**.
3. Drag & drop `datasets.zip` into the upload modal.
4. Enter any dataset title (for example, `te-q-transformer-data`) and click **Create**.
   > *Kaggle automatically extracts the zip under `/kaggle/input/<dataset-title>/`.*

---

### Step 2: Import & Configure Notebook

1. In Kaggle, click **+ Create** &rarr; **New Notebook**.
2. Go to **File** &rarr; **Import Notebook** &rarr; Select and upload whichever experiment notebook you wish to review from the `notebooks/` directory:
   - `notebooks/benchmark/baseline_comparison_nasa.ipynb` &mdash; 10 Baselines vs. TE-Q-Transformer benchmark
   - `notebooks/statistical_significance.ipynb` &mdash; Wilcoxon signed-rank tests & block bootstrap
   - `notebooks/explainability.ipynb` &mdash; Integrated Gradients attributions & physics sanity check
   - `notebooks/Calce.ipynb` &mdash; CALCE zero-shot / cross-cell evaluation
   - `notebooks/ablation/ablation_study.ipynb` &mdash; 5 architectural ablation configurations
   - `notebooks/generalization/generalization_study.ipynb` &mdash; Unseen-cell & temporal extrapolation
3. In the right-hand settings panel:
   - Click **+ Add Input** &rarr; Select **Your Datasets** &rarr; Click **+** to attach your uploaded dataset.
   - Set **Internet** to **ON** (required for `pip install pennylane`).
   - Set **Accelerator** to **GPU P100** or **GPU T4 x2** (Quantum circuit simulation executes significantly faster with GPU acceleration; CPU is also supported).

---

### Step 3: Run All & View / Download Results

1. Click **Run All** (`Shift + Enter`).
2. The notebook will automatically:
   - Auto-detect dataset files, checkpoints, and dependencies without any manual path adjustments.
   - Install PennyLane if not already present.
   - Execute inference, evaluations, and visualizations.
   - Display all publication-quality comparison figures and numerical metric tables directly in the output cells.
   - Package all high-resolution figures (`.png`, `.pdf`), metric tables (`.csv`), and summary logs into a single downloadable **`*_Results.zip`** in `/kaggle/working/`.
3. In the right-hand panel under **Output** (`/kaggle/working`), click the three dots next to the generated zip file and select **Download**.

| Experiment Notebook | Generated Output Zip | Output Artifacts Included |
|---|---|---|
| `baseline_comparison_nasa.ipynb` | `NASA_Baseline_Comparison_Results.zip` | Benchmark figures, 11-model comparison CSV, paper summary table |
| `statistical_significance.ipynb` | `Statistical_Significance_Results.zip` | $p$-values table, bootstrap distribution plots, confidence intervals |
| `explainability.ipynb` | `Explainability_Results.zip` | Attribution heatmaps, temperature sensitivity curves, sanity checks |
| `Calce.ipynb` | `CALCE_Evaluation_Results.zip` | CALCE degradation curves, cross-chemistry transfer tables |
| `ablation_study.ipynb` | `Ablation_Study_Results.zip` | Component ablation radar/bar charts, ablation metrics CSV |
| `generalization_study.ipynb` | `Generalization_Study_Results.zip` | Unseen-cell trajectory plots, temporal extrapolation reports |

---

## Dataset Specifications & Experimental Partitions

The repository processes battery cycle measurements into fixed-length arrays matching a strict input contract.

### Expected Tensor Format

- **Input Array Shape**: `[B, 512, 4]` (Batch size $B$, sequence length 512, 4 channels).
- **Target Array Shape**: `[B]` (Scalar float32 representing $SOH_k = C_k / C_0$).
- **Channel Ordering**:
  - Index 0: **Voltage** ($V$)
  - Index 1: **Current** ($A$)
  - Index 2: **Temperature** ($T_C$ in degrees Celsius)
  - Index 3: **Normalized Time** ($t_{\text{norm}} \in [0, 1]$ within the discharge cycle)

### Feature Scaling & Non-Leakage Contract

- Channels 0, 1, and 3 are normalized using `sklearn.preprocessing.MinMaxScaler` fitted **exclusively on the training cell partition**.
- Channel 2 (**Temperature**) is kept strictly unscaled in degrees Celsius throughout preprocessing and data loading. This ensures that physical temperatures can be converted to Kelvin ($T_K = T_C + 273.15$) for Arrhenius kinetic equations.
- Test cell cycles are transformed using the fitted training scaler without refitting.

### NASA Ames Battery Dataset

- **Source**: NASA Prognostics Center of Excellence (PCoE) Battery Data Set.
- **Chemistry**: Commercial 18650 LiCoO₂ / graphite cylindrical cells (nominal capacity 2.0 Ah).
- **Stored Location**: `datasets/NASA/processed/` (`{cell}_X.npy`, `{cell}_soh.npy`).
- **Partitioning**:
  - **Training Pool (660 cycles)**: `B0005`, `B0006`, `B0007` (24°C), `B0029`, `B0030`, `B0031` (4°C), and first 70% of `B0053` (44°C, 37 cycles).
  - **Held-Out Unseen Cells (171 cycles)**: `B0018` (24°C, 132 cycles) and `B0032` (4°C, 39 cycles).
  - **Temporal Extrapolation (16 cycles)**: Final 30% of `B0053` (44°C).

### CALCE Battery Dataset

- **Source**: Center for Advanced Life Cycle Engineering (CALCE), University of Maryland.
- **Chemistry**: Prismatic LiCoO₂ CS2 series (`CS2_35`, `CS2_36`, `CS2_37`, `CS2_38`, nominal capacity 1.1 Ah).
- **Stored Location**: `datasets/CALCE/processed/` (`{cell}_X_unscaled.npy`, `{cell}_soh.npy`, `{cell}_cycle.npy`).
- **Partitioning**:
  - **Training Split (2,309 cycles)**: `CS2_35`, `CS2_36`, and the first 70% of `CS2_37`.
  - **Held-Out Unseen Cell Test Split (966 cycles)**: `CS2_38`.
  - **Temporal Extrapolation Test Split (275 cycles)**: Remaining 30% of `CS2_37` (`CS2_37_test`).

---

## Expected Outputs

When executing training and evaluation scripts or notebooks in this repository, the following artifacts are produced:

- **Model Predictions**: Tensors of shape `[B]` representing scalar SOH estimates ($C_k / C_0$) corresponding to each input cycle window.
- **Training Logs**: Output logs reporting Mean Squared Error (MSE) training loss per epoch, learning rate scheduler updates, and periodic validation metrics.
- **Evaluation Metrics**: Regression metric summaries computed via `src.eval.metrics.calculate_metrics`:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - Mean Absolute Percentage Error (MAPE)
  - Coefficient of Determination ($R^2$)
  - Maximum Error (MaxE)
- **Model Checkpoints**: Serialized PyTorch state dictionaries (`.pt` files) saved to designated checkpoint directories.
- **Visualizations**: Generated matplotlib plots showing training loss trajectories, predicted versus measured SOH degradation curves across cycles, and error distributions.

---

## Reproducibility Notes

To ensure consistent experimental reproduction across environments:

- **Random Seed Handling**: Notebooks implement a centralized seeding routine (`seed_everything(seed=42)`) that fixes seeds for Python `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and sets `torch.backends.cudnn.deterministic = True`.
- **Consistent Tensor Dimensions**: All loaders and models enforce the `[B, 512, 4]` tensor contract and channel order `[Voltage, Current, Temperature_C, Normalized_Time]`.
- **Data Non-Leakage Contract**: `MinMaxScaler` parameters are computed exclusively on designated training partitions. Test sets are transformed using the fitted scaler and are never used to compute minimum or maximum bounds.
- **Preserved Physical Temperature**: Channel 2 is never rescaled; it remains in degrees Celsius so that absolute temperature ($T_K = T_C + 273.15$) is substituted into the Arrhenius equation.
- **Deterministic Dependencies**: Package versions are specified in `requirements.txt`. For quantum simulations, PennyLane state-vector routines are executed on fixed seed states.

---

## Dataset Citations

#### NASA Ames Battery Dataset
```bibtex
@misc{saha2007battery,
  title={Battery Data Set},
  author={Saha, Bhaskar and Goebel, Kai},
  year={2007},
  publisher={NASA Ames Prognostics Center of Excellence (PCoE)},
  url={https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/}
}
```

#### CALCE Battery Dataset
```bibtex
@misc{calce2011battery,
  title={CALCE Battery Data Set},
  author={{Center for Advanced Life Cycle Engineering (CALCE)}},
  year={2011},
  publisher={University of Maryland},
  url={https://calce.umd.edu/battery-data}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
