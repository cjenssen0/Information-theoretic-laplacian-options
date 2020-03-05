#!/bin/bash

# Copy scripts
scp -P 30564 './keca_options.sh' root@springfield.uit.no:~/

# Run file on springfield cluster
kubectl job run keca_options.yml
