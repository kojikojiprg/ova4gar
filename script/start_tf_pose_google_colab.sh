apt-get install swig
apt-get install ffmpeg

git clone https://github.com/ildoonet/tf-pose-estimation.git

dir=/content/tf-pose-estimation

pip install -r $dir/requirements.txt

bash $dir/models/graph/cmu/download.sh

cd $dir/tf_pose/pafprocess/
swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace

pip install -U pip
pip uninstall -y tensorflow
pip install tensorflow-gpu==1.15.3
