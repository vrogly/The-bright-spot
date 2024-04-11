#!/bin/bash
folderName=cropped-$(date "+%Y%m%d%H%M%S")
mkdir $folderName
for var in $(find *)
do
	echo $var
	ffmpeg -loglevel panic -i $var -vf "crop=350:350:849:849"  $folderName/cropped-$var
done
