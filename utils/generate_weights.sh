#!/usr/bin/env bash

set -e
set -x

cd /opt/yolov5

mkdir -p /var/app/weights/yolov5n/
mkdir -p /var/app/weights/yolov5s/
mkdir -p /var/app/weights/yolov5n6/
mkdir -p /var/app/weights/yolov5s6/

curl -o /var/app/weights/yolov5n/yolov5n.pt -L https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt
curl -o /var/app/weights/yolov5n6/yolov5n6.pt -L https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n6.pt
curl -o /var/app/weights/yolov5s/yolov5s.pt -L https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
curl -o /var/app/weights/yolov5s6/yolov5s6.pt -L https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s6.pt

python3 /opt/yolov5/gen_wts.py -w /var/app/weights/yolov5n/yolov5n.pt -o /var/app/weights/yolov5n/yolov5n.wts
python3 /opt/yolov5/gen_wts.py -w /var/app/weights/yolov5n6/yolov5n6.pt -o /var/app/weights/yolov5n6/yolov5n6.wts
python3 /opt/yolov5/gen_wts.py -w /var/app/weights/yolov5s/yolov5s.pt -o /var/app/weights/yolov5s/yolov5s.wts
python3 /opt/yolov5/gen_wts.py -w /var/app/weights/yolov5s6/yolov5s6.pt -o /var/app/weights/yolov5s6/yolov5s6.wts

# yolov5n
python3 export.py --weights yolov5n.pt --include torchscript &&
        mv /opt/yolov5/yolov5n.torchscript.pt /var/app/weights/yolov5n/yolov5n.torchscript.cpu.640.pt
python3 export.py --weights yolov5n.pt --include torchscript --device 0 &&
        mv /opt/yolov5/yolov5n.torchscript.pt /var/app/weights/yolov5n/yolov5n.torchscript.gpu.640.pt
python3 export.py --weights yolov5n.pt --include torchscript --device 0 --half &&
        mv /opt/yolov5/yolov5n.torchscript.pt /var/app/weights/yolov5n/yolov5n.torchscript.gpu.half.640.pt

# yolov5s
python3 export.py --weights yolov5s.pt --include torchscript &&
        mv /opt/yolov5/yolov5s.torchscript.pt /var/app/weights/yolov5s/yolov5s.torchscript.cpu.640.pt
python3 export.py --weights yolov5s.pt --include torchscript --device 0 &&
        mv /opt/yolov5/yolov5s.torchscript.pt /var/app/weights/yolov5s/yolov5s.torchscript.gpu.640.pt
python3 export.py --weights yolov5s.pt --include torchscript --device 0 --half &&
        mv /opt/yolov5/yolov5s.torchscript.pt /var/app/weights/yolov5s/yolov5s.torchscript.gpu.half.640.pt

# yolov5n6
python3 export.py --weights yolov5n6.pt --include torchscript --imgsz 1280 &&
        mv /opt/yolov5/yolov5n6.torchscript.pt /var/app/weights/yolov5n6/yolov5n6.torchscript.cpu.1280.pt
python3 export.py --weights yolov5n6.pt --include torchscript --device 0 --imgsz 1280 &&
        mv /opt/yolov5/yolov5n6.torchscript.pt /var/app/weights/yolov5n6/yolov5n6.torchscript.gpu.1280.pt
python3 export.py --weights yolov5n6.pt --include torchscript --device 0 --half --imgsz 1280 &&
        mv /opt/yolov5/yolov5n6.torchscript.pt /var/app/weights/yolov5n6/yolov5n6.torchscript.gpu.half.1280.pt

# yolov5s6
python3 export.py --weights yolov5s6.pt --include torchscript --imgsz 1280 &&
        mv /opt/yolov5/yolov5s6.torchscript.pt /var/app/weights/yolov5s6/yolov5s6.torchscript.cpu.1280.pt
python3 export.py --weights yolov5s6.pt --include torchscript --device 0 --imgsz 1280 &&
        mv /opt/yolov5/yolov5s6.torchscript.pt /var/app/weights/yolov5s6/yolov5s6.torchscript.gpu.1280.pt
python3 export.py --weights yolov5s6.pt --include torchscript --device 0 --half --imgsz 1280 &&
        mv /opt/yolov5/yolov5s6.torchscript.pt /var/app/weights/yolov5s6/yolov5s6.torchscript.gpu.half.1280.pt
