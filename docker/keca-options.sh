#!/bin/sh

# Go to correct dir
cd keca_options_rl

# Install depenencies
echo 'Cloning and installing'
git clone -b jonas https://github.com/cjenssen0/Information-theoretic-laplacian-options.git
cd Information-theoretic-laplacian-options/

pip install rlglue

#echo 'Runnning laplace tabular'
#python laplace_tabular.py
echo 'Runnning keca tabular'
python keca_tabular.py
#echo 'Runnning kpca tabular'
#python kernel_matrix_tabular.py
