docker run -it -d --network host --gpus=all --restart=always --name detector detector_api:latest bash -c "python /workspace/api/libs/connection/server.py"
