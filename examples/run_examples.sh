#!/bin/bash
set -ex

pip install sentence-transformers peft openai cohere voyageai

exit_status=0

run_command() {
    status=0
    "$@" || status=$?
    if [ $status -ne 0 ]; then
        echo "Error with '$*': exited with $status"
        exit_status=1 # Update the exit status if the command failed
    fi
}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
dir_path=$(dirname "${BASH_SOURCE[0]}")
run_command python ${dir_path}/excelformer.py --epochs 1
# run_command python ${dir_path}/llm_embedding.py --epochs 1
run_command python ${dir_path}/mercari.py --epochs 1
run_command python ${dir_path}/revisiting.py --epochs 1
run_command python ${dir_path}/tab_transformer.py --epochs 1
run_command python ${dir_path}/tabnet.py --epochs 1
run_command python ${dir_path}/transformers_text.py --epochs 1
run_command python ${dir_path}/trompt.py --epochs 1
run_command python ${dir_path}/tuned_gbdt.py
run_command python ${dir_path}/tutorial.py --epochs 1

exit $exit_status
