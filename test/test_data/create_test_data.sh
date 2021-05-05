#!/bin/bash
cd "${0%/*}" || exit
#------------------------------------------------------------------------------

# definition of path variables
source_folder="./test_cases/"
run_folder="./run/"

# base name of singularity image
image_base_name="openfoam-v2006"

# check if image exists
if [ ! -f ${image_base_name}.sif ]; then
  echo -e "Error: could not find singularity image \033[1m${image_base_name}.sif\033[0m"
  echo "To build the image, run"
  echo ""
  echo "sudo singularity build ${image_base_name}.sif ${image_base_name}.def"
  echo ""
  exit 1
fi

# create run directory if necessary
if [ ! -d $run_folder ]
then
    echo ""
    echo "Creating run directory $run_folder"
    echo "mkdir -p $run_folder"
    $(mkdir -p $run_folder)
    echo ""
fi

# run test cases
run_test_case(){
    case=$1
    if [ -d $run_folder$case ]
    then
        echo -e "test case \033[1m$case\033[0m already exists; skipping case..."
    else
        echo "Copying test case $case"
        copy="cp -r $source_folder$case $run_folder$case"
        echo $copy
        $copy
        echo "Running Allrun script"
        run="singularity run openfoam-v2006.sif run $run_folder$case"
        echo $run
        $run
    fi
}

test_cases=$(ls ${source_folder})
for case in ${test_cases[@]}; do
    run_test_case $case
done

#------------------------------------------------------------------------------
