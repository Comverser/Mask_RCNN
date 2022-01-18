# prerequisites
cuda 10.1
cudnn 7.6

# install
```bash
conda create -n maskrcnn python=3.7.3
conda activate maskrcnn
pip3 install -r requirements.txt
python3 setup.py install
```
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

# error
permission error
```bash
sudo chmod -R 777 /<target>
```
