# openpose installation
参考
* https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md
* https://stella-log.hatenablog.com/entry/2019/01/11/174950
* https://qiita.com/takahyon/items/197250194edf2d6ec041

1. git clone

```
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
```

2. フォルダ移動

```
cd ~/openpose
```

3. コメントアウトされている部分の # を削除

```
vim scripts/osx/install_deps.sh
```

4. CMake をインストール

```
brew cask install cmake
```

5. bash でスクリプトを起動し必要なパッケージをインストール

```
bash scripts/osx/install_deps.sh
```

6. caffe をインストール

```
brew install caffe
```

7. 空のフォルダ ```build/``` を作成

8. CMake の GUI を起動する

```
cmake-gui
```

9. 設定は [こちら](https://stella-log.hatenablog.com/entry/2019/01/11/174950) の CMakeでビルド設定を参照

10. ```build/``` に移動してビルドする。

```
cd build
```

```
make -j`sysctl -n hw.logicalcpu`
```

11. vecLibが見つからなくてエラーが起きた場合

```~/openpose/build/caffe/src/openpose_lib-build/CMakeCache.txt``` の ```//vecLib include directory``` を変更する。

```
vecLib_INCLUDE_DIR:PATH=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/
```

12. インストールする。

```
sudo make install
```


# openpose reinstallation

1. openpose を アンインストール

```
sudo make uninstall
```

2. ```~/openpose/build/``` を削除し、空のフォルダを作成

3. CMake GUI 上で ```File -> Delete Cache``` を実行

4. [openpose installation](#openpose-installation) 手順 9. から繰り返す


# Python で openpose を import

1. ```~/openpose/build/python/openpose/``` をフォルダごとコピー

2. ```.venv/lib/python3.7/site-packages/``` にペースト

3. ```openpose/``` 内の ```CMakeFiles/``` , ```cmake_install.cmake``` , ```MakeFile``` は削除しても大丈夫

4. 呼び出しには以下を記述

``` python
from openpose import pyopenpose
```
