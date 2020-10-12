# OCR Indonsia

This repositories is for prototype OCR's API deployment.

# Testing Environment

```
ROG Strix Hero II GL504GV
NVIDIA GeForce RTX 2060
=========================
Ubuntu 18.04
=========================
python 3.8.3
cuda 10.2
pytorch 1.5.1
opencv 4.3.0
```


# Preparation

1. Download and Install the mandatory packages, using script below : <br>

```console
conda install python
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install opencv-contrib-python
pip install pandas
```

2. Build Detectron2 localy, you can refer to [this link](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) but here we will give you a quick installation : <br>

### Windows
```console
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Linux
```console
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.5/index.html
```

3. Clone this repository `git clone https://gitlab.com/aempat/proto_ocr.git`

4. Download our pretrained model [here](https://drive.google.com/drive/folders/1wRdZsX_yvNcyGUSKwHvqoaU3OO8R1s9l?usp=sharing) and [here](https://drive.google.com/drive/folders/1g3HuX2fHTsGc7CTrPikbQge705ABt59B), and put it in `model_weights` folder.

`if you havn't downloaded the model yet. We make the model can be downloaded automatically, when you call the module`
  
```
├── model_weights
│   ├── faster_rcnn_R_101_FPN_3x_NIK_NOSIM.pth
│   │   ...
│   └── model_final_ktp_sim-fix_rgb_rotated_X101_custom_mapper.pth
└── Examples.ipynb
```

# Demo

We provide streamlit apps for this OCR application, follow this 2 steps in terminal inside this repo:
1. `pip install streamlit`
2. `streamlit run streamlit_ocr.py`

go to `http://{your ip or localhost}:8501`, select document type, upload photo then get the result

# Detail Example

you can run our [notebook](/Examples.ipynb) for detailed example. Please be aware with `list_cuda` parameters on cfg, the default is set on 4 GPUs. if you only have one GPU, set list_cuda to `0` in every cfg.

# Contact

[Ziyad S Fawwazi] <br>
[Hervind] <br>
[Andreas Mulya R]

