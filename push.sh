#!/bin/sh
#
# example usage of script:
#   bash push.sh lab_oi_nigeria nigeria_acled temporal/v1
#
# where:
#   - lab_oi_nigeria is the project name and corresponds to directory in /sciclone/aiddata10/REU/projects
#   - nigeria_acled is the basename of a json file in the settings dir
#   - test/v1 is the prefix used to create define path for push (creates subdir if "/" is used). _nigeria_acled will be appended to the prefix
#
# this example will result in files being pushed to:
#   /sciclone/aiddata10/REU/projects/lab_oi_nigeria/test/v1_nigeria_acled
#   with all relevant scripts updated to use the "settings/nigeria_acled.json" path
#


project=$1
settings=$2
dir_prefix=$3

dir=${dir_prefix}_${settings}

if [[ $dir == "" ]]; then
    echo "Must specify job dir"
    exit 1
fi

src=/home/${USER}/git/geo-cnn
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
pushr utils


# -----------------------------------------------------------------------------

cp s1_jobscript     tmp_s1_jobscript
cp s2_jobscript     tmp_s2_jobscript
cp s3_s1_jobscript  tmp_s3_s1_jobscript
cp s3_s2_jobscript  tmp_s3_s2_jobscript
cp s4_jobscript     tmp_s4_jobscript

echo "python /sciclone/aiddata10/REU/projects/"${project}"/"${dir}"/s1_main.py" >> s1_jobscript
echo "mpirun --mca mpi_warn_on_fork 0 --map-by node python-mpi /sciclone/aiddata10/REU/projects/"${project}"/"${dir}"/s2_main.py" >> s2_jobscript
echo "python /sciclone/aiddata10/REU/projects/"${project}"/"${dir}"/s3_s1_predict.py" >> s3_s1_jobscript
echo "python /sciclone/aiddata10/REU/projects/"${project}"/"${dir}"/s3_s2_predict.py" >> s3_s2_jobscript
echo "python /sciclone/aiddata10/REU/projects/"${project}"/"${dir}"/s4_main.py" >> s4_jobscript

push s1_jobscript
push s2_jobscript
push s3_s1_jobscript
push s3_s2_jobscript
push s4_jobscript

cp tmp_s1_jobscript s1_jobscript
cp tmp_s2_jobscript s2_jobscript
cp tmp_s3_s1_jobscript s3_s1_jobscript
cp tmp_s3_s2_jobscript s3_s2_jobscript
cp tmp_s4_jobscript s4_jobscript

rm tmp_s1_jobscript
rm tmp_s2_jobscript
rm tmp_s3_s1_jobscript
rm tmp_s3_s2_jobscript
rm tmp_s4_jobscript

# -----------------------------------------------------------------------------

sfiles=(
    build_grid.py
    s1_main.py
    s1_merge.py
    s2_main.py
    s2_merge.py
    s3_s1_predict.py
    s3_s2_predict.py
    s4_main.py

)

    # s1_validation_nigeria.py
    # s4_validation.py
    # s4_validation_nigeria.py

for i in "${sfiles[@]}"*; do
    cp ${i} tmp_${i}
    sed -i 's+json_path = "settings/settings_example.json"+json_path = "settings/'${settings}'.json"+' ${i}
    push ${i}
    cp tmp_${i} ${i}
    rm tmp_${i}
done
