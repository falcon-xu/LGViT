# LGViT

Official PyTorch implementation of "LGViT: Dynamic Early Exiting for Accelerating Vision Transformer" (**ACM MM 2023**)

## Usage

First, clone the repository locally:

```bash
git clone https://github.com/lostsword/LGViT
```

Then, install PyTorch and [transformers 4.26.0](https://github.com/huggingface/transformers)

```bash
conda create -n lgvit python=3.9.13
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.26.0 datasets==2.9.0 evaluate==0.4.0 timm==0.6.13 wandb==0.14.2 ipykernel scikit-learn
```

Enter the `scripts` folder to execute the scripts for training and evaluation

```bash
cd ./scripts
```

- **train_base_deit.sh / train_base_swin.sh**

  This is for fine-tuning base models.

- **train_baseline_deit.sh / train_baseline_swin.sh**

  This is for fine-tuning *1st stage LGViT* models and baseline models.

- **train_distillation_deit.sh / train_distillation_swin.sh**

  This is for fine-tuning *2nd stage  LGViT models*.

- **eval_highway_deit.sh / eval_highway_swin.sh**

​		This is for evaluating fine-tuned models.

Before running the script, modify the `path` and `model_path` in the script to be appropriate.

### Training

To fine-tune a ViT backbone, run:

```bash
source train_base_deit.sh
```

To fine-tune a LGViT models, run:

```bash
source train_baseline_deit.sh
source train_distillation_deit.sh
```

### Evaluation

To evaluate a fine-tuned ViT, run:

```bash
source eval_highway_deit.sh
```



### Some Hyperparameters Settings


- Exiting points settings

|          | ViT-EE |          Others           |          LGViT          |
| :------: | :----: | :-----------------------: | :---------------------: |
| ViT-B/16 |  [6]   | [1,2,3,4,5,6,7,8,9,10,11] |   [4,5,6,7,8,9,10,11]   |
|  DeiT-B  |  [6]   | [1,2,3,4,5,6,7,8,9,10,11] |   [4,5,6,7,8,9,10,11]   |
|  Swin-B  |  [12]  |  [4,7,10,13,16,19,22,23]  | [4,7,10,13,16,19,22,23] |

Other hyperparameters are are kept unchanged from the original baselines.



## Acknowledgments

This repository is built upon the [transformers](https://github.com/huggingface/transformers) and [DeeBERT](https://github.com/castorini/DeeBERT) library. Thanks for these awesome open-source projects!

## Citation

If you find our work or this code useful, please consider citing the corresponding paper:
```
@article{xu2023lgvit,
  title={LGViT: Dynamic Early Exiting for Accelerating Vision Transformer},
  author={Xu, Guanyu and Hao, Jiawei and Shen, Li and Hu, Han and Luo, Yong and Lin, Hui and Shen, Jialie},
  journal={arXiv preprint arXiv:2308.00255},
  year={2023}
}
```