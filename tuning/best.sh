#!/usr/bin/bash

echo "Time: $(date)"
python yolov5/train.py --data data.yaml --project runs --single-cls --device 0 --hyp hyps/3.yaml --epochs 10 --batch-size 15 --optimizer Adam
