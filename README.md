# Lossless Neural Compression

To start the virtual env, run:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For training and testing, download and unzip the provided PNG dataset into the `data` directory. To download other datasets, run:
```bash
mkdir data
cd data
wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d tiny-imagenet-200
curl -L -C - -o lhq-1024.zip https://www.kaggle.com/api/v1/datasets/download/dimensi0n/lhq-1024
unzip lhq-1024.zip -d lhq-1024
```

To evaluate the model, run:
```bash
sbatch train.sh
sbatch reconstruct.sh
python analyze.py
```

### Notes
- Before running the scripts, change the `config.yaml` file to select the desired dataset path and experiments directory