docker run -it -d --network host --gpus=all --restart=always --name detector detector_api:latest bash -c "export PYTHONPATH=$PYTHONPATH:/workspace/api/libs && export PYTHONPATH=$PYTHONPATH:/workspace/api/libs/yolov7 && cd /workspace/api && python run_server.py"

