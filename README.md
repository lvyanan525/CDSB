### Official implementation of CDSB: Accelerating Connectomics workflow via Content-Decoupled Schrödinger Bridge



#### Getting Started

##### Installation

- Clone this repo:

```
git clone https://github.com/lvyanan525/CDSB.git
cd CDSB
```

- Install [PyTorch](http://pytorch.org/) and other dependencies. We provide an essential `requirements.txt` based on the project's dependency tree. You can install them via pip:

```
pip install -r requirements.txt
```

##### Data Preparation

Organize your datasets into separate directories for each modality. For example, prepare one folder containing your source images (e.g., Light Microscopy) and another for your target images (e.g., Electron Microscopy).

```
datasets/
├── your_tissue_dataset/
│   ├── train/
│   │   ├── LM/         # Source modality (e.g., Light Microscopy)
│   │   └── EM/         # Target modality (e.g., Electron Microscopy)
│   └── test/
│       ├── LM/
│       └── EM/
```
You can also download the dataset in  https://pan.baidu.com/s/1tLWu2t9fWmoJobyz9eLwyg?pwd=fswk 提取码: fswk

#### Training

To train the CDSB model on your dataset, run the main training script. The training pipeline is handled by PyTorch Lightning, making it easy to scale.

```
bash train.sh
```

or

```
python train.py \
    --lr 5e-5 \
    --lr_brightness 5e-5 \
    --batch_size 8 \
    --image_size $IMAGE_SIZE \
    --max_steps $MAX_STEPS \
    --val_every_n_batches $VAL_EVERY_N_BATCHES \
    --save_every_n_steps $SAVE_EVERY_N_STEPS \
```

#### Inference

Once the model is trained, you can perform inference from your source directory (LM) to generate predictions of the target directory (EM).

```
python inference.py --emsb_ckpt checkpoints/best_model.ckpt \
                    --input_dir ./datasets/your_tissue_dataset/test/LM \
                    --save_dir ./results/ \
                    --nfe 20 \
```

or you can download the .pt in ./ckpts at https://pan.baidu.com/s/1G_rt73NtB0eWnrV6WTejsA?pwd=eqjg 提取码: eqjg 

