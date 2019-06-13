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
push load_data.py
push load_survey_data.py
push main.py
push second_stage_model.py
push model_prep.py
push merge_outputs.py

for i in settings_*; do
    push $i
done

# -----------------------------------------------------------------------------

cp s1_jobscript tmp_s1_jobscript
cp s2_jobscript tmp_s2_jobscript

echo "python /sciclone/aiddata10/REU/projects/"${project}"/"${dir}"/main.py" >> s1_jobscript
echo "mpirun --mca mpi_warn_on_fork 0 --map-by node python-mpi /sciclone/aiddata10/REU/projects/"${project}"/"${dir}"/second_stage_model.py" >> s2_jobscript

push s1_jobscript
push s2_jobscript

cp tmp_s1_jobscript s1_jobscript
cp tmp_s2_jobscript s2_jobscript

rm tmp_s1_jobscript
rm tmp_s2_jobscript

# -----------------------------------------------------------------------------

cp main.py tmp_main.py
cp second_stage_model.py tmp_second_stage_model.py
cp merge_outputs.py tmp_merge_outputs.py
cp build_surface_grid.py tmp_build_surface_grid.py

sed -i 's+json_path = "settings/settings_example.json"+json_path = "settings/'${settings}'.json"+' main.py
sed -i 's+json_path = "settings/settings_example.json"+json_path = "settings/'${settings}'.json"+' second_stage_model.py
sed -i 's+json_path = "settings/settings_example.json"+json_path = "settings/'${settings}'.json"+' merge_outputs.py
sed -i 's+json_path = "settings/settings_example.json"+json_path = "settings/'${settings}'.json"+' build_surface_grid.py

push main.py
push second_stage_model.py
push merge_outputs.py
push build_surface_grid.py

cp tmp_main.py main.py
cp tmp_second_stage_model.py second_stage_model.py
cp tmp_merge_outputs.py merge_outputs.py
cp tmp_build_surface_grid.py build_surface_grid.py

rm tmp_main.py
rm tmp_second_stage_model.py
rm tmp_merge_outputs.py
rm tmp_build_surface_grid.py
