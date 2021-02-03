FROM pangwei/wilds_src:1.0

WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt && \
    # Install torch-scatter and torch-geometric with the PyTorch and CUDA version specified
    # All baseline experiments in the paper were run on Python 3.8.5 and CUDA 10.1.
    pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html && \
    pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html && \
    pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html && \
    pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html && \
    pip install torch-geometric \
