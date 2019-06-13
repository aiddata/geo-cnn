#!/bin/sh

project=$1
settings=$2
dir_prefix=$3

dir=${dir_prefix}_${settings}

if [[ $dir == "" ]]; then
    echo "Must specify job dir"
    exit 1
fi

src=/home/userz/git/geo-cnn
base="/sciclone/aiddata10/REU/projects/${project}"

dst=$base/$dir

mkdir -p $dst

push() {
    rm $dst/$1
    cp $src/$1 $dst/$1
}

pushr() {
    rm -r $dst/$1
    cp -r $src/$1 $dst/$1
}

pushr scripts
pushr settings

push data_prep.py
push create_grid.py
push runscript.py
push resnet.py
push vgg.py
push load_ntl_data.py
push load_landsat_data.py
push load_survey_data.py
push model_prep.py

for i in settings_*; do
    push $i
done

# -----------------------------------------------------------------------------

cp s1_jobscript tmp_s1_jobscript
cp s2_jobscript tmp_s2_jobscript

echo "python /sciclone/aiddata10/REU/projects/"${project}"/"${dir}"/s1_main.py" >> s1_jobscript
echo "mpirun --mca mpi_warn_on_fork 0 --map-by node python-mpi /sciclone/aiddata10/REU/projects/"${project}"/"${dir}"/s2_main.py" >> s2_jobscript

push s1_jobscript
push s2_jobscript

cp tmp_s1_jobscript s1_jobscript
cp tmp_s2_jobscript s2_jobscript

rm tmp_s1_jobscript
rm tmp_s2_jobscript

# -----------------------------------------------------------------------------

cp s1_main.py tmp_s1_main.py
cp s2_main.py tmp_s2_main.py
cp s2_merge.py tmp_s2_merge.py
cp s3_build_grid.py tmp_s3_build_grid.py

sed -i 's+json_path = "settings/settings_example.json"+json_path = "settings/'${settings}'.json"+' s1_main.py
sed -i 's+json_path = "settings/settings_example.json"+json_path = "settings/'${settings}'.json"+' s2_main.py
sed -i 's+json_path = "settings/settings_example.json"+json_path = "settings/'${settings}'.json"+' s2_merge.py
sed -i 's+json_path = "settings/settings_example.json"+json_path = "settings/'${settings}'.json"+' s3_build_grid.py

push s1_main.py
push s2_main.py
push s2_merge.py
push s3_build_grid.py

cp tmp_s1_main.py s1_main.py
cp tmp_s2_main.py s2_main.py
cp tmp_s2_merge.py s2_merge.py
cp tmp_s3_build_grid.py s3_build_grid.py

rm tmp_s1_main.py
rm tmp_s2_main.py
rm tmp_s2_merge.py
rm tmp_s3_build_grid.py
