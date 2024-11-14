#!/bin/bash

# chain id and python file
chain_id="$1"
py_file="run_1"
if [[ $# -gt 1 ]]; then
	py_file="$2"
fi

# create directory
mkdir -p "$py_file"

(

# source conda
#source ${HOME}/opt/anaconda3/etc/profile.d/conda.sh

OSTYPE="$(uname -s)"

if [[ $OSTYPE == "Darwin" ]]; then
	source ~/.bash_profile
	source ${HOME}/opt/anaconda3/etc/profile.d/conda.sh
else
	source ~/.bashrc
	source ${HOME}/anaconda3/etc/profile.d/conda.sh
fi

# activate conda
conda activate confen

# run
echo "pwd: $(pwd)"
echo "py file: ${py_file}"
echo " "
echo "Running chain ${chain_id}"
echo " "
python  "${py_file}.py" \
		--chain_id="${chain_id}"

) | tee "${py_file}/chain_${chain_id}.log"
