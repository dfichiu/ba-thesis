#!/bin/sh

while getopts y: flag
do
    case "${flag}" in
        y) filename_yaml=${OPTARG};;
    esac
done
echo "Currently running $filename_yaml";

# Basename: Remove the suffix ".yaml" from  the file name.
# sed: Remove "experiment-" prefix. 
experiment_number=$(basename "$filename_yaml" .yaml | sed 's/experiment-//')

#echo "Extracted experiment number $experiment_number"

# Print the extracted experiment number from the YAML file's name.
# echo "The value of x is: $experiment_number"


# Provide a YAML file from which the experiment parameter values are read.
# Run the experiment.
papermill experiment_template.ipynb "run_experiment-$experiment_number".ipynb -f $filename_yaml

# Create .html from run experiment notebook as.
jupyter nbconvert --to html --output-dir="results" --template full --no-input run_experiment-$experiment_number.ipynb


# Remove run notebook.
rm "run_experiment-$experiment_number".ipynb