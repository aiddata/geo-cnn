#!/bin/sh

dir=$1

if [[ $dir == "" ]]; then
    echo "Must specify job dir"
    exit 1
fi

src=/home/userz/git/geo-cnn
base="/sciclone/aiddata10/REU/projects/mcc_tanzania"

dst=$base/$dir

mkdir -p $dst

push() {
    cp $src/$1 $dst/$1
}

push prepare_survey_data.py
push data_prep.py
push create_grid.py
push runscript.py
push resnet.py
push vgg.py
push load_data.py
push load_survey_data.py
push main.py
push second_stage_model.py

for i in settings_*; do
    push $i
done

cp s1_jobscript tmp_s1_jobscript
cp s2_jobscript tmp_s2_jobscript

echo "python /sciclone/aiddata10/REU/projects/mcc_tanzania/"${dir}"/main.py" >> s1_jobscript
echo "mpirun --mca mpi_warn_on_fork 0 --map-by node -np 80 python-mpi /sciclone/aiddata10/REU/projects/mcc_tanzania/"${dir}"/second_stage_model.py" >> s2_jobscript

push s1_jobscript
push s2_jobscript

cp tmp_s1_jobscript s1_jobscript
cp tmp_s2_jobscript s2_jobscript

rm tmp_s1_jobscript
rm tmp_s2_jobscript
