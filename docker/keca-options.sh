#!/bin/sh

# Install depenencies
git clone -b jonas https://github.com/cjenssen0/Information-theoretic-laplacian-options.git
cd Information-theoretic-laplacian-options/

pip install rlglue

python laplace_tabular.py
python keca_tabular.py
python kernel_matrix_tabular.py
