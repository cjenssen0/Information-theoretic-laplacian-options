#!/bin/bash

# Copy results
scp -P 30564 -r root@springfield.uit.no:~/keca_options_rl/Information-theoretic-laplacian-options/data_files/average_return_keca.npy ./results/
scp -P 30564 -r root@springfield.uit.no:~/keca_options_rl/Information-theoretic-laplacian-options/data_files/average_return_laplace.npy ./results/
scp -P 30564 -r root@springfield.uit.no:~/keca_options_rl/Information-theoretic-laplacian-options/data_files/average_return_kernel_matrix.npy ./results/
