# (公式)openpose
## installation

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
cd openpose
```

3. CMake をインストール

```
brew cask install cmake
```

4. コメントアウトされている部分の # を削除
   ```pip``` でインストールされる numpy opencv-python は仮想環境上にインストールするので削除

```
vim scripts/osx/install_deps.sh
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

10. ```Configure``` 終了後に ```BUILD_PYTHON``` が表示されているので、チェックを入れてもう一度 ```Configure``` と ```Generate``` を実行する。

11. ```build/``` に移動してビルドする。

```
cd build
```

```
make -j`sysctl -n hw.logicalcpu`
```

12. vecLibが見つからなくてエラーが起きた場合

```./openpose/build/caffe/src/openpose_lib-build/CMakeCache.txt``` の ```//vecLib include directory``` を変更する。

```
vecLib_INCLUDE_DIR:PATH=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/
```

もう一度 11. を実行する。

13. ```./openpose/build/``` から ```./``` に戻り、 ```./openpose/build/python/openpose/``` をフォルダごと仮想環境にコピーする

```
cd ../../
cp ./openpose/build/python/openpose/ .venv/lib/python3.7/site-packages/```
```

呼び出しには以下を記述

``` python
from openpose import pyopenpose
```


## reinstallation

1. ```~/openpose/build/``` を削除し、空の ```build/``` フォルダを作成

2. CMake GUI 上で ```File -> Delete Cache``` を実行

3. [openpose installation](#openpose-installation) 手順 9. から繰り返す


# tf-pose-estimation (使用していない)
## poetry installation

poetry で パッケージ管理する

```
brew install poetry
```

プロジェクトディレクトリ内で Poetry を適用

```
poetry init
```

## installation

start.sh で tf-pose-estimation をインストール

```
zsh start.sh
```

```tf-pose-estimation/tf_pose/``` を Python のファイルを管理するディレクトりにコピーする
