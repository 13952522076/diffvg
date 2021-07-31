#!/usr/bin/env bash

for filename in $(ls images_resized_2/)
do

	echo $filename
	python rendering.py --target images_resized_2/$filename --use_blob --folder rendering_resize
done;
