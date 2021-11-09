FROM python:3.6

WORKDIR /root

ADD protos/ protos/

ADD ecosys/ ecosys/
ADD setup.cfg .
ADD pyproject.toml .
ADD README.md .

ADD service/ service/

COPY docker/coordinator/requirements.txt .

RUN pip3 install -r requirements.txt

RUN python3 -m grpc_tools.protoc -Iprotos --python_out=protos/. --grpc_python_out=protos/. ./protos/ecosys.proto
RUN python3 -m build && pip3 install dist/*.tar.gz

RUN pip3 cache purge

# ENTRYPOINT python service/coordinator.py