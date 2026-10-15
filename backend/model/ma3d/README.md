# MA3D-Net: Multi-modal Adaptive 3D Morphable Model Modulation Network for Robust Facial Expression Recognition

## EMOCA Feature Extraction

Extract 3DMM facial expression features using EMOCA for all datasets.
The extracted features will be used as input for downstream facial expression recognition experiments.

### Requirements

Before running EMOCA extraction:

- Clone and install the official [EMOCA](https://github.com/radekd91/emoca) repository.
- Follow the installation instructions provided by the EMOCA repository.

### Dataset Preparation

Prepare the following datasets:

- **[RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset)**
- **[FERPlus](https://github.com/microsoft/FERPlus)**
- **[CAER-S](https://caer-dataset.github.io/)**
- **[CheoFaMo](https://data.mendeley.com/datasets/2h6hxwkbwn/2)**

For each dataset, run `compute_stats` defined inside each dataset class to calculate the normalization statistics for 3DMM.

### Extraction Process

For each dataset:

1. Load the dataset using the corresponding dataset class.
2. Run EMOCA inference on all images to extract FLAME parameters.
3. EMOCA will automatically generate the FLAME/3DMM parameter files in `.npy` format. These outputs are already in the required format for MA3D, so no additional conversion, modification, or post-processing is needed.


The generated EMOCA outputs can be directly used as 3DMM inputs for MA3D training and evaluation.

### Expected Output Structure

Ensure your dataset is organized as follows:

```
- Datasets/raf-db/
    - test/
        - 1/
            - test_0002_aligned/
                - test_0002_aligned.jpg
                - *.npy (3DMM extracted from EMOCA)
    - train/
```

## Environment Setup

### Option 1: Conda

```bash
conda create -n ma3d python=3.8
conda activate ma3d
pip install -r requirements.txt
```

### Option 2: Docker

Use the provided Dockerfile to build the environment.

## Pretrained Models

Download pretrained weights (Image backbone and Landmark backbone) from [here](https://drive.google.com/drive/folders/1X9pE-NmyRwvBGpVzJOEvLqRPRfk_Siwq?usp=sharing). Place the downloaded files into the `checkpoints` folder.

Expected structure:

```
- checkpoints/
    - ir50.pth
    - mobilefacenet_model_best.pth.tar
```


## Hardware Information

All experiments were conducted on a machine with an NVIDIA GeForce GTX 1660 (6GB VRAM).

For this hardware configuration, a batch size of **8** was used during training. 
The batch size can be adjusted depending on the available GPU memory. If your GPU has more VRAM, you may increase the batch size; otherwise, reduce it to avoid out-of-memory errors.


## Training

To train on a specific dataset (e.g., RAF-DB):


```bash
python train.py --seed 42--data_type RAF-DB --batch_size 8 --resume_name checkpoints/rafdb_best.pth
```

The `batch_size` can be scaled according to the available GPU resources. Using a larger batch size may lead to better performance. Training logs are saved in the `log` directory. Running the training multiple times is recommended to obtain optimal results.
## Testing

If you want to use our pretrained model, you can download the best checkpoint [here](https://drive.google.com/drive/folders/10tltNSB009VoalxTo46IUMpQhPANudhF?usp=drive_link) and place it into the `checkpoints/` directory.

To evaluate on a specific dataset (e.g., RAF-DB), run:

```bash
python test.py --seed 42 --data_type RAF-DB --ckpt_path checkpoints/rafdb_best.pth
```

## Ablation Study Checkpoints

The checkpoints used for ablation experiments are provided separately.

You can download the ablation checkpoints [here](https://drive.google.com/drive/folders/10ysC_2tfkfZTyalj-nvX96FtAsCC8uCl?usp=sharing) and place them into the `checkpoints/ablation/` directory.

Note that each ablation experiment requires the corresponding changes in the model architecture or training configuration. Before evaluating an ablation checkpoint, please ensure that the model settings and training parameters are consistent with the specific ablation design.

For example, to evaluate an ablation model:

```bash
python test.py --seed 42 --data_type RAF-DB --ckpt_path checkpoints/ablation/baseline.pth
```
## Visualization

You can visualize model outputs such as:

- t-SNE visualizations  
- Attention maps  
- Radar charts  

Run the scripts in the `scripts/` directory.
