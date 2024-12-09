#!/bin/bash 

for experiment in experiment-*.py; do 
    base_name="${experiment%.py}" 
    python "$experiment" > "${base_name}.txt" 
done 