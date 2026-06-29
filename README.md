# MorphST: Morphology-Guided Graph Learning and Multi-Modal Fusion for Spatial Domain Identification


## Requirements
create environment:
```bash
conda create -n MorphST python=3.8.13
conda activate MorphST
```

Install the required packages:
```bash
pip install -r requirements.txt
```


## Usage

### Raw Data Preparation

Place the raw spatial transcriptomics data (e.g., DLPFC) in the folder'data/'.

### Data Preprocessing

For the DLPFC dataset:

```bash
python DLPFC_generate_data.py
```

For other datasets, modify the dataset name in `HBC_generate_data.py` and run:

```bash
python HBC_generate_data.py
```

### Model Training and Testing

For the DLPFC dataset:

```bash
python DLPFC_test.py
```

For other datasets, modify the dataset name in `HBC_test.py` and run:

```bash
python HBC_test.py
```

### Results
 
The prediction results and evaluation metrics are saved in the `result/` directory.


