# wewantCNNs_project
To get started we recommend quickly creating a venv with all the dependencies from 'Requirements.txt'
## Create virtual environment
**Using `virtualenv`:**
```
virtualenv /path/to/where/you/want/your_env_name
```
**Using `venv`:**
```
python3 -m venv /path/to/new/virtual/environment
```

## Activate virtual environment

```
source /path/to/your_env_name/bin/activate
```

## Install dependencies

```
pip install -r Requirements.txt
```

## Test the code

Run the streamlit frontend with 

```
streamlit run streamlit_app/main.py
```

## Coming Soon:

 - Better sample dataset with files already as spectograms in HDF5 format for faster access and therefore training.
 - Working NF model showcase.
 - Module for inference 
#### Coming later:

- Latent space interface for to work with trained models during inference.

## Credits:

Autoencoder Model inspired by: https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/
Dataset:  Jesse Engel, Cinjon Resnick, Adam Roberts, Sander Dieleman, Douglas Eck,
Karen Simonyan, and Mohammad Norouzi. "Neural Audio Synthesis of Musical Notes
with WaveNet Autoencoders." 2017.
