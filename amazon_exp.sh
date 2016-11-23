#!/usr/bin/env bash

for (( c=12; c<24; c++ ))
do  
	echo "Experiments for DA Task $c times"
	python exp_domainreg_da.py AMT $c AMT_res/
done
