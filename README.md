


# Environment Setup

Please be aware that this code is meant to be run with Python 3.10.___. Download the packages from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```
1. Run `download_abide.py` to download the raw data.

We use the command below:

```bash
python download_abide.py
```

2. To show the demographic Information of ABIDE I.

```bash
python pheno_info.py
```

3. Run `prepare_data.py` to compute the correlation. Then we can get the hdf5 files.

```bash
python prepare_data.py --folds=10 --whole cc200 aal ez
```

4. Using Stacked Sparse Denoising Autoencoder (SSDAE) to perform Multi-atlas Deep Feature Representation, and using multilayer perceptron (MLP) and ensemble learning to classify the ASD and TC.

```bash
rm ./data/models/*mlp*
python nn.py --whole cc200 aal ez
```  

5. Evaluating the MLP model on test dataset.
```bash
python nn_evaluate.py --whole cc200 aal ez
```