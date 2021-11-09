#!/bin/bash

VERSION=$(head -n 1 VERSION)

docker build -t serving/env:${VERSION} -f docker/env/Dockerfile .
docker build -t serving/coordinator:${VERSION} -f docker/coordinator/Dockerfile .