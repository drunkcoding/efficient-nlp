#!/bin/bash

LIB=ecosys
PROTOS=protos

python -m grpc_tools.protoc -I${PROTOS} --python_out=${PROTOS}/. --grpc_python_out=${PROTOS}/. ${PROTOS}/${LIB}.proto

sed -i 's/import ecosys_pb2/from \. import ecosys_pb2/g' ${PROTOS}/ecosys_pb2_grpc.py

python3 -m build
pip3 install dist/*.tar.gz

export PYTHONPATH=${PYTHONPATH}:$(pwd)/protos