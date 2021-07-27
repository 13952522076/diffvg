#!/usr/bin/env bash

for filename in $(ls images_resized/)
do

	echo $filename
	python rendering.py --target images_resized/$filename --use_blob --folder rendering_resize
done;
