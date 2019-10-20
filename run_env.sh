docker build -t super_res .
docker run --rm \
    --runtime=nvidia \
    -it -v $PWD:/home \
    super_res:latest \
    bash
