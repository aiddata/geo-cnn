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
push model_predict.py

for i in settings_*; do
    push $i
done

# -----------------------------------------------------------------------------

cp s1_jobscript tmp_s1_jobscript
cp s2_jobscript tmp_s2_jobscript
cp s3_s1_jobscript tmp_s3_s1_jobscript
cp s3_s2_jobscript tmp_s3_s2_jobscript

echo "python /sciclone/aiddata10/REU/projects/"${project}"/"${dir}"/s1_main.py" >> s1_jobscript
echo "mpirun --mca mpi_warn_on_fork 0 --map-by node python-mpi /sciclone/aiddata10/REU/projects/"${project}"/"${dir}"/s2_main.py" >> s2_jobscript
echo "python /sciclone/aiddata10/REU/projects/"${project}"/"${dir}"/s3_s1_predict.py" >> s3_s1_jobscript
echo "python /sciclone/aiddata10/REU/projects/"${project}"/"${dir}"/s3_s2_predict.py" >> s3_s2_jobscript

push s1_jobscript
push s2_jobscript
push s3_s1_jobscript
push s3_s2_jobscript

cp tmp_s1_jobscript s1_jobscript
cp tmp_s2_jobscript s2_jobscript
cp tmp_s3_s1_jobscript s3_s1_jobscript
cp tmp_s3_s2_jobscript s3_s2_jobscript

rm tmp_s1_jobscript
rm tmp_s2_jobscript
rm tmp_s3_s1_jobscript
rm tmp_s3_s2_jobscript

# -----------------------------------------------------------------------------

sfiles=(
    s1_main.py
    s2_main.py
    s2_merge.py
    s3_build_grid.py
    s3_s1_predict.py
    s3_s2_predict.py
)

for i in "${sfiles[@]}"*; do
    cp ${i} tmp_${i}
    sed -i 's+json_path = "settings/settings_example.json"+json_path = "settings/'${settings}'.json"+' ${i}
    push ${i}
    cp tmp_${i} ${i}
    rm tmp_${i}
done
