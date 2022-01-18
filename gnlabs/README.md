# install
```bash
git clone https://github.com/Comverser/Mask_RCNN.git
cd Mask_RCNN
conda create -n maskrcnn --file gnlabs/working-gpu.txt
conda activate maskrcnn
python3 setup.py install
```
jupyter lab 
```bash
conda install -c conda-forge jupyterlab
jupyter server --generate-config
jupyter server --generate-config
conda install -c conda-forge nb_conda_kernels
```

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
