#!/usr/bin/env bash

for filename in $(ls images2/)
do

	echo $filename
	python rendering.py --target images2/$filename --use_blob
done;
