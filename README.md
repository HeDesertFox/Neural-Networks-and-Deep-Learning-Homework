# Neural-Networks-and-Deep-Learning-Homework

Assignments for the Neural Network and Deep Learning course at Fudan University (Lecturer: Zhang Li), completed by Yihan He and Wentao Lv.

## Folder Structure

- **mid_term**: Contains midterm assignments.
- **final**: Contains final assignments.

Each folder includes several tasks.

## Midterm Assignments

### Task 1
Detailed instructions for the fine-tuning task can be found in the `mid_term/task1_finetuning/main_notebook.ipynb` file. Please follow the instructions and run each cell sequentially.

### Task 2
For the object detection task, you need to have mmdetection installed. Simply run:

```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
mim install mmdet
```

After installing successfully, run the following:

```
cd mmdetection
python tools/train.py \
    ${CONFIG_FILE} \
    --auto-scale-lr \
    [optional arguments]
```
Where the config files are provided in our project.

## Final Assignments

### Task 1

For pretraining the ResNet18 with SimCLR, follow the instructions in `pretraining_notebook.ipynb`. Note that you may need to manually create some folders that do not exist.

### Task 2

Detailed instructions for the Transformer-CNN comparison task can be found in the `final/task2_transformer_vs_CNN/test_notebook.ipynb` file. Please follow the instructions and run each cell sequentially.

### Task 3

For training NeRF, you need to have NeRFstudio installed. Run the following commands:

```
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip
pip uninstall torch torchvision functorch tinycudann
```

Note that torch needs to be installed, with a version of either 2.1.2 or 2.0.1. Then run:

```
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio
```

Then download the dataset from our provided BaiduNetDisk URL and place it at `{DATA_DIR}`. After that, run:

```
conda install colmap
ns-process-data {images, video} --data {DATA_PATH} --output-dir {PROCESSED_DATA_DIR}
ns-train nerfacto --data {PROCESSED_DATA_DIR} --vis tensorboard
```