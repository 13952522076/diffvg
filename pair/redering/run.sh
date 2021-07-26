#!/usr/bin/env bash

for filename in $(ls images/)
do

	echo $filename
	python rendering.py --target images/$filename --use_blob
done;
