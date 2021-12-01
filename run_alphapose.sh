#!/bin/bash

# 引数確認
if [ "$#" -ne 2 ]
then
    echo "bash run_alphapose.sh [VIDEO_FILE_PATH] [CUDA_DEVICE_NUM (0 ~ 4)]"
    exit 1
fi
readonly video_path=$1
readonly cuda=$2

# ビデオファイルパス
video_full_path='/tmp/sh076018/mishare/Research Projects/yokoyama/'$video_path
echo "video_path: "$video_full_path

# 出力ディレクトリ
video_dir="${video_path%.*}"  # delete ".mp4"
video_dir="${video_dir#video/}"  # delete "video/"
out_dir='/tmp/sh076018/mishare/Research Projects/yokoyama/data/'$video_dir'/video/'
echo "out_dir: "$out_dir

# AlphaPoseのモデルパス
model_dir='/tmp/sh076018/mishare/Research Projects/yokoyama/model/alphapose/'
cfg_path=$model_dir'256x192_w32_lr1e-3.yaml'
checkpoint_path=$model_dir'hrnet_w32_256x192.pth'

# Run alphapose
cd ./alphapose
echo "CUDA_VISIBLE_DEVICES="$cuda
CUDA_VISIBLE_DEVICES=$cuda python3 scripts/demo_inference.py \
    --cfg "$cfg_path" \
    --checkpoint "$checkpoint_path" \
    --outdir "$out_dir" \
    --eval \
    --video "$video_full_path" \
    --save_video \
    --detbatch 40 \
    --posebatch 10000 \
    --qsize 4096 \
    --vis_fast \
    # --sp \

# jsonファイル置き換え
json_path="${out_dir%video/}json/"
mkdir -p "${json_path}"
mv "${out_dir}alphapose-results.json" "${json_path}"