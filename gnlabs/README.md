# install
```bash
# Create a conda environment
conda create -n maskrcnn --file gnlabs/working-gpu.txt
conda activate maskrcnn

# Install Mask RCNN
git clone https://github.com/Comverser/Mask_RCNN.git
cd Mask_RCNN
python3 setup.py install
```
jupyter lab 
```bash
pip install jupyterlab
jupyter notebook password
```

conda env
```bash
conda install -c conda-forge nb_conda_kernels
```

## requirements
hardware compatible to 
- cudnn 7.6.4
- cuda 10.1_0.tar.bz2

# demo
```bash
mkdir datasets
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip -P datasets/
unzip -q datasets/balloon_dataset.zip -d datasets/
rm datasets/balloon_dataset.zip
```
```bash
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_balloon.h5
```
```bash
conda activate maskrcnn
jupyter-lab --ip 0.0.0.0 --port 8888 --no-browser
```
