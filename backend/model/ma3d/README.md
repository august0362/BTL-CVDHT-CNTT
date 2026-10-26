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

`models/MA3D.py` loads them as `checkpoints/ir50_o1.pth` and `checkpoints/mobilefacenet_model_best.pth_o1.tar`
(relative to the working directory), so also keep a copy under those names.


## Hardware Information

All experiments were conducted on a machine with an NVIDIA GeForce GTX 1660 (6GB VRAM).

For this hardware configuration, a batch size of **8** was used during training. 
The batch size can be adjusted depending on the available GPU memory. If your GPU has more VRAM, you may increase the batch size; otherwise, reduce it to avoid out-of-memory errors.


## Training

To train on a specific dataset (e.g., RAF-DB):


```bash
python train.py --seed 42 --data_type RAF-DB --batch_size 8 --resume_name checkpoints/rafdb_best.pth
```

The `batch_size` can be scaled according to the available GPU resources. Using a larger batch size may lead to better performance. Training logs are saved in the `log` directory. Running the training multiple times is recommended to obtain optimal results.

### Loss options

`--loss` selects the training objective:

- `mace` (default): CE + 2 × label-smoothing CE for the first 10 epochs (warm-up), then `MarginAwareCELoss` only,
  as described in the paper.
- `bmace`: `BalancedMarginAwareCELoss`. For the first 10 epochs it is label-smoothed CE (ε = 2/15, the same
  smoothing as CE + 2 × LSCE(0.2)) on logits shifted by `tau × log(class prior)`; the MACE-style margin weights
  (detached from the gradient) are then ramped in over 5 epochs. The prior shift is only applied during training;
  evaluation uses the raw logits.

`--tau` sets the strength of the class-prior shift. Larger values favour minority classes more and can lower overall
accuracy sharply, so choose it on a held-out part of the training set instead of the test set:

```bash
# 1) Choose tau on a stratified 15% of the training set (the test set is not used).
for tau in 0 0.1 0.25 0.5; do
  python train.py --data_type CheoFaMo --num_classes 6 --loss bmace --tau $tau --val_split 0.15 --seed 42 \
    --log_file cheofamo_tau$tau.txt --resume_name cheofamo_tau$tau.pth
done
# Keep the tau with the highest "Best validation macro-F1" at the end of its log.

# 2) Train on the full training set with that tau, over several seeds.
for seed in 0 42 123 3407 9999; do
  python train.py --data_type CheoFaMo --num_classes 6 --loss bmace --tau 0.25 --seed $seed \
    --log_file cheofamo_bmace_s$seed.txt --resume_name cheofamo_bmace_s$seed.pth
done
```

Validation also reports balanced accuracy and macro-F1 (`Val_BAcc` and `Val_MF1` in the log). `--seed` fixes the
Python, NumPy and PyTorch seeds (cuDNN can still make GPU runs differ slightly), and `--split_seed` (default 0) fixes
the held-out split so that every tau is compared on the same samples.

### Comparing the losses on CheoFaMo

`scripts/compare_losses.py` checks the setup, then trains `--loss mace` and `--loss bmace` one after the other with
the same seed on 85% of the CheoFaMo training set, validates both on the held-out 15%, and prints a table:

```bash
python -m scripts.compare_losses --check     # data, weights, packages and a time estimate
python -m scripts.compare_losses             # 30 epochs per loss (about 3 h each on a GTX 1660 Ti)
python -m scripts.compare_losses --summary   # table of the finished runs
```

It expects:

```
- checkpoints/
    - ir50_o1.pth, mobilefacenet_model_best.pth_o1.tar   (see Pretrained Models)
- Datasets/CheoFamo/
    - Metadata.json
    - cheofamo_stats.npz                     (written by --compute_stats if missing)
    - Frames/<video>_<character>_<apex>/     e.g. 01_02_00123/
        - the segment image
        - exp.npy, detail.npy, pose.npy, shape.npy, tex.npy   (EMOCA; shapes 50, 128, 6, 100, 50)
```

Finished runs are skipped when the script is started again, and an interrupted run starts over. `--taus 0.1 0.25 0.5`
adds more settings and `--epochs 100` uses the full schedule. Note that `compute_cheofamo_stats` averages over all
segments, test segments included.

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
