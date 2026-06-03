# MorphST: Morphology-Guided Graph Learning and Multi-Modal Fusion for Spatial Domain Identification


## create environment
`conda create -n MorphST python=3.8.13`

`pip install -r requirements.txt`

## Usage

### Raw Data Preparation

Place the raw spatial transcriptomics data (e.g., DLPFC) in the folder ***data***.

### Data Preprocessing

Run ***DLPFC_generate_data.py*** to preprocess the raw DLPFC data:

`python DLPFC_generate_data.py`

### Model Training and Testing

Run ***DLPFC_test.py*** to train and test the MorphST model:

`python DLPFC_test.py`

### Results
 
The results, including predictions and evaluation metrics, are saved in the folder ***result***.


