# SCOUT

Implementation for the paper **"SCOUT: Graph-Guided Discovery of Exceptional Subgroups with Gaussian Mixtures"**.

---

## Setup

### Perceptive Graph Construction

Use `perceptive_construction.ipynb` to construct perceptive graphs.

Use `synthetic.ipynb` to create perceptive graphs for the synthetic datasets.

---

## Running Experiments

### Main Experiments

```bash
python experiments.py {name} {pocket_size} {measure}
```

`name` is the dataset name as defined in `datasets/meta_data.py`.

### Synthetic Runtime Experiments

```bash
python synthetic.py {name} {size}
```

`name=N` for cardinality scaling, `name=C` for descriptor dimensionality scaling.

### Ablation Study

```bash
python ablation.py {name} {pocket_size}
```

`name` is the dataset name as defined in `datasets/meta_data.py`.

### Case Study

Use `case_study.ipynb` to prepare the banknote dataset for SCOUT.

```bash
python case_study.py {pocket_size}
```

Runs SCOUT on the prepared case study dataset.

---

## Visualisations

Use `experiment_explorer.ipynb` to generate visualisations and tables from experiment results.

Use `case_study.ipynb` for case study visualisations.