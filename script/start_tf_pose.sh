brew install swig
brew install ffmpeg

git clone https://github.com/ildoonet/tf-pose-estimation.git

poetry add cython numpy opencv-python tensorflow==1.15.3
poetry add `cat tf-pose-estimation/requirements.txt`

. .venv/bin/activate

cd tf-pose-estimation
zsh ./models/graph/cmu/download.sh

cd tf_pose/pafprocess/
swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace

deactivate

cd ../../../
