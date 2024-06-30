# Neural-Networks-and-Deep-Learning-Homework

Assignments for the Neural Network and Deep Learning course at Fudan University (Lecturer: Zhang Li), completed by Yihan He and Wentao Lv.

## Folder Structure

- **mid_term**: Contains midterm assignments.
- **final**: Contains final assignments.

Each folder includes several tasks.

## Midterm Assignments

### Task1
Detailed instructions for the fine-tuning task can be found in the `mid_term/task1_finetuning/main_notebook.ipynb` file. Please follow the instructions and run each cell sequentially.

### Task2
For object detection task, it requires mmdetection installed, simply

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

After installing successfully, run the following

```
cd mmdetection
python tools/train.py \
    ${CONFIG_FILE} \
    --auto-scale-lr \
    [optional arguments]
```
Where the config files are provided in our project.
## Final Assignments
### Task1

For pretraining the ResNet18 with SimCLR, follow instruction of `pretraining_notebook.ipynb`, note that user shall manually create some of the folders which doesn't exist.

### Task2

Detailed instructions for the Transformer-CNN comparing task can be found in the `final/task2_transformer_vs_CNN/test_notebook.ipynb` file. Please follow the instructions and run each cell sequentially.

### Task3

For training nerf, it requires NeRFstudio to be installed.

```
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip
pip uninstall torch torchvision functorch tinycudann
```

Note that torch needed to be installed, the version shall be either 2.1.2 or 2.0.1.

```
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```
```
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio
```

Then download the dataset from our provided BaiduNetDisk url and place it at `{DATA_DIR}`, then
`conda install colmap`
`ns-process-data {images, video} --data {DATA_PATH} --output-dir {PROCESSED_DATA_DIR}`
`ns-train nerfacto --data {PROCESSED_DATA_DIR} --vis tensorboard`