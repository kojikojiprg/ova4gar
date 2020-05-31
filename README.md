# openpose installation
参考
* https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md
* https://stella-log.hatenablog.com/entry/2019/01/11/174950
* https://qiita.com/takahyon/items/197250194edf2d6ec041

git clone

```
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
```

フォルダ移動

```
cd ~/openpose
```

コメントアウトされている部分の # を削除

```
vim scripts/osx/install_deps.sh
```

cmake をインストール

```
brew cask install cmake
```

bash でスクリプトを起動し必要なパッケージをインストール

```
bash scripts/osx/install_deps.sh
```

caffe をインストール

```
brew install caffe
```

Cmake の GUI を起動する

```
cmake-gui
```

設定は [こちら](https://stella-log.hatenablog.com/entry/2019/01/11/174950) の CMakeでビルド設定を参照

```build/``` に移動してビルドする。

```
cd build
```

```
make -j`sysctl -n hw.logicalcpu`
```

vecLibが見つからなくてエラーが起きた場合

```~/openpose/build/caffe/src/openpose_lib-build/CMakeCache.txt``` の ```//vecLib include directory``` を変更する。

```
vecLib_INCLUDE_DIR:PATH=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/
```

インストールする。

```
sudo make install
```


# Python で openpose を import

```~/openpose/build/python/openpose/``` をフォルダごとコピー

```.venv/lib/python3.7/site-packages/``` にペースト

```openpose/``` 内の ```CMakeFiles/``` , ```cmake_install.cmake``` , ```MakeFile``` は削除しても大丈夫

呼び出しには以下を記述

``` python
from openpose import pyopenpose
```
