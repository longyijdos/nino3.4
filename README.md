# nino3.4
The pytorch implementation of our paper

## Environment Setup

### Install CUDA

Make sure to install the appropriate version of CUDA on your system. You can download the CUDA installation package from the NVIDIA website and follow the official guide for installation.

### Install PyTorch

PyTorch is an open-source machine learning library for deep learning tasks. It provides rich tools and functions to simplify the process of building and training neural networks. You can install PyTorch using pip with the following command:

```bash
pip install torch torchvision torchaudio
```

### Install Project Dependencies

To install project dependencies, navigate to the root directory of your project where the requirements.txt file is located and run the following command:

```bash
pip install -r requirements.txt
```
## Data Preparation

### Download

Run the script with the following command:

```bash
cd data_preprocess
python download.py --name [dataset_name] --save_path [save_path]
```

Replace [dataset_name] with the name of the dataset you want to download (cmip6, errstv6, godas, or soda). You can also specify the save path using the --save_path argument.
Repeat the command for each dataset to ensure you have downloaded all required datasets.

### Preprocess

Run the script with the following command:

```bash
cd data_preprocess
python preprocess.py --name [dataset_name] --data_path [data_path] --save_path [save_path]
```

Replace [dataset_name] with the name of the dataset you want to preprocess (cmip6, errstv6, godas, or soda).
Use the --data_path argument to specify the path to the dataset.
Use the --save_path argument to specify the path where the preprocessed data will be saved.
Repeat the above steps to preprocess all datasets.
