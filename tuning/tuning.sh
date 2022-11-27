#!/usr/bin/bash

echo "Time: $(date)"
python yolov5/train.py --data data.yaml --project runs --single-cls --device 0 --hyp hyps/1.yaml --epochs 3 --batch-size 15 --optimizer SGD

echo "Time: $(date)"
python yolov5/train.py --data data.yaml --project runs --single-cls --device 0 --hyp hyps/2.yaml --epochs 3 --batch-size 15 --optimizer SGD

echo "Time: $(date)"
python yolov5/train.py --data data.yaml --project runs --single-cls --device 0 --hyp hyps/3.yaml --epochs 3 --batch-size 15 --optimizer SGD

echo "Time: $(date)"
python yolov5/train.py --data data.yaml --project runs --single-cls --device 0 --hyp hyps/4.yaml --epochs 3 --batch-size 15 --optimizer SGD


echo "Time: $(date)"
python yolov5/train.py --data data.yaml --project runs --single-cls --device 0 --hyp hyps/1.yaml --epochs 3 --batch-size 15 --optimizer Adam

echo "Time: $(date)"
python yolov5/train.py --data data.yaml --project runs --single-cls --device 0 --hyp hyps/2.yaml --epochs 3 --batch-size 15 --optimizer Adam

echo "Time: $(date)"
python yolov5/train.py --data data.yaml --project runs --single-cls --device 0 --hyp hyps/3.yaml --epochs 3 --batch-size 15 --optimizer Adam

echo "Time: $(date)"
python yolov5/train.py --data data.yaml --project runs --single-cls --device 0 --hyp hyps/4.yaml --epochs 3 --batch-size 15 --optimizer Adam
