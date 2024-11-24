#!/bin/bash

for i in 0.1 0.3 0.8;
do
	python3 Q1.py $i 45
done

for i in 15 30 45 60 75;
do
	python3 Q1.py 0.1 $i
done
