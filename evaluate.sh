MODEL_DIR="lstm_rezoom_2017_04_15_13.43"
GPU=1

python evaluate.py --weights output/$MODEL_DIR/save.ckpt-50000 --test_boxes data/webcam/test_boxes.json --gpu $GPU >> result/$MODEL_DIR/log_050000.txt
python evaluate.py --weights output/$MODEL_DIR/save.ckpt-60000 --test_boxes data/webcam/test_boxes.json --gpu $GPU >> result/$MODEL_DIR/log_060000.txt
python evaluate.py --weights output/$MODEL_DIR/save.ckpt-70000 --test_boxes data/webcam/test_boxes.json --gpu $GPU >> result/$MODEL_DIR/log_070000.txt
python evaluate.py --weights output/$MODEL_DIR/save.ckpt-80000 --test_boxes data/webcam/test_boxes.json --gpu $GPU >> result/$MODEL_DIR/log_080000.txt
python evaluate.py --weights output/$MODEL_DIR/save.ckpt-90000 --test_boxes data/webcam/test_boxes.json --gpu $GPU >> result/$MODEL_DIR/log_090000.txt
python evaluate.py --weights output/$MODEL_DIR/save.ckpt-100000 --test_boxes data/webcam/test_boxes.json --gpu $GPU >> result/$MODEL_DIR/log_100000.txt
python evaluate.py --weights output/$MODEL_DIR/save.ckpt-110000 --test_boxes data/webcam/test_boxes.json --gpu $GPU >> result/$MODEL_DIR/log_110000.txt
python evaluate.py --weights output/$MODEL_DIR/save.ckpt-120000 --test_boxes data/webcam/test_boxes.json --gpu $GPU >> result/$MODEL_DIR/log_120000.txt
python evaluate.py --weights output/$MODEL_DIR/save.ckpt-130000 --test_boxes data/webcam/test_boxes.json --gpu $GPU >> result/$MODEL_DIR/log_130000.txt
python evaluate.py --weights output/$MODEL_DIR/save.ckpt-140000 --test_boxes data/webcam/test_boxes.json --gpu $GPU >> result/$MODEL_DIR/log_140000.txt
python evaluate.py --weights output/$MODEL_DIR/save.ckpt-150000 --test_boxes data/webcam/test_boxes.json --gpu $GPU >> result/$MODEL_DIR/log_150000.txt
python evaluate.py --weights output/$MODEL_DIR/save.ckpt-160000 --test_boxes data/webcam/test_boxes.json --gpu $GPU >> result/$MODEL_DIR/log_160000.txt
python evaluate.py --weights output/$MODEL_DIR/save.ckpt-170000 --test_boxes data/webcam/test_boxes.json --gpu $GPU >> result/$MODEL_DIR/log_170000.txt
python evaluate.py --weights output/$MODEL_DIR/save.ckpt-180000 --test_boxes data/webcam/test_boxes.json --gpu $GPU >> result/$MODEL_DIR/log_180000.txt
python evaluate.py --weights output/$MODEL_DIR/save.ckpt-190000 --test_boxes data/webcam/test_boxes.json --gpu $GPU >> result/$MODEL_DIR/log_190000.txt
python evaluate.py --weights output/$MODEL_DIR/save.ckpt-200000 --test_boxes data/webcam/test_boxes.json --gpu $GPU >> result/$MODEL_DIR/log_200000.txt
python evaluate.py --weights output/$MODEL_DIR/save.ckpt-250000 --test_boxes data/webcam/test_boxes.json --gpu $GPU >> result/$MODEL_DIR/log_250000.txt
python evaluate.py --weights output/$MODEL_DIR/save.ckpt-300000 --test_boxes data/webcam/test_boxes.json --gpu $GPU >> result/$MODEL_DIR/log_300000.txt
