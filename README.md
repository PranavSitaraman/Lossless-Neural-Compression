# Lossless Neural Compression

### Pranav Sitaraman, Gavin Ye, Alex Todoran

To start the virtual env, run:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For training and testing, download and unzip the provided PNG dataset [here](https://drive.google.com/file/d/1dgd5M3arBknNX1oekJPeBa4551Bd5Rkv/view?usp=drive_link) into the `data` directory.

To train and evaluate the model, run:
```bash
torchrun --nproc_per_node 4 train.py
torchrun --nproc_per_node 4 test.py
```

### Notes
- Before running the scripts, change the `config.yaml` file to select the desired dataset path and experiments directory.
- Current tests partition the dataset into 3000 initial images of training data followed by 500 unseen images used for testing.