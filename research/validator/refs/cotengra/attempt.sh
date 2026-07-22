#!/bin/bash
D=/Users/liujinguo/rcode/omeco/research/validator/refs/cotengra
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$D/venv/bin/python" "$D/run_cotengra.py" "$1" "$2" "$3"
