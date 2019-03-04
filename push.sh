#!/bin/sh

src=/home/userz/git/geo-cnn
dst="/sciclone/aiddata10/REU/projects/mcc_tanzania"

cp $src/create_grid.py $dst
cp $src/runscript.py $dst
cp $src/jobscript $dst
cp $src/resnet.py $dst
cp $src/load_data.py $dst
cp $src/data_prep.py $dst
cp $src/main.py $dst
cp $src/second_stage_model.py $dst
