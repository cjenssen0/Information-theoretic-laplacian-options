#!/bin/bash

# Copy scripts
scp -P 30564 './keca-options.sh' root@springfield.uit.no:~/keca_options_rl/

# Run file on springfield cluster
kubectl job run keca_options.yml
