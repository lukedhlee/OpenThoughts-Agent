#!/bin/bash
# usage: submit_arm.sh <arm> [extra sbatch args]
A=$1; shift
sbatch --parsable --job-name snowball_bench_$A --export=ALL,ARMFILE=/e/project1/transfernetx/lee27/code/snowball/bench/arms/$A.env "$@" /e/project1/transfernetx/lee27/code/snowball/bench/bench_serve.sbatch
