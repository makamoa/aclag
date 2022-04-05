#!/bin/bash --login

set -euo pipefail

# create the conda environment
export ENV_PREFIX=$PWD/env
mamba env create --prefix $ENV_PREFIX --file environment.yml --force

# install a package $PWD/src in 'development mode'
conda develop -p $PWD/env $PWD/src/
