#!/bin/sh

## I. Dataset sample statistics
echo "=> Currently computing and saving dataset sample statistics";
ipython -c "%run dataset_statistics.ipynb"


## II. Attention visualization
echo "=> Currently visualizing attention and saving visualizations";
sleep 1s
echo "The entire attention landscapes cannot be saved since they are Javascript objects. Please run 'visualize_attention.ipynb' if you want to see them.";
ipython -c "%run visualize_attention.ipynb"


## III. Subsequence construction
echo "=> Currently constructing subsequences from the in-sample inference sentences";
# Parametrize notebooks: w/ & w/o stop words.
papermill construct_subsequences.ipynb constructed_subsequences_without_stopwords.ipynb -p remove_stopwords True
papermill construct_subsequences.ipynb constructed_subsequences_with_stopwords.ipynb -p remove_stopwords False

# Create .html files from run experiment notebooks.
jupyter nbconvert --to html --output-dir="results/html" --template full --no-input constructed_subsequences_without_stopwords.ipynb
jupyter nbconvert --to html --output-dir="results/html" --template full --no-input constructed_subsequences_with_stopwords.ipynb

# Remove parametrized notebooks.
rm constructed_subsequences_without_stopwords.ipynb
rm constructed_subsequences_with_stopwords.ipynb


## IV. Sliding window n-gram method
echo "=> Currently running the sliding window n-gram method experiment";
jupyter nbconvert --to html --output-dir="results/html" --template full --no-input experiment_sliding_window.ipynb


## V. Transformer attention experiments
echo "=> Currently running minig Transformer attention experiments";
# Parametrize notebooks: experiment number & filename.
papermill test_memory_template.ipynb test_memory_3.ipynb -p filename "2023-09-10 17-02-12-946618.pkl" -p experiment_no 3
papermill test_memory_template.ipynb test_memory_1.ipynb -p filename "2023-09-10 13-01-52-622408.pkl" -p experiment_no 1


# Create .html files from run experiment notebooks.
jupyter nbconvert --to html --output-dir="results/html" --template full --no-input test_memory_3.ipynb
jupyter nbconvert --to html --output-dir="results/html" --template full --no-input test_memory_1.ipynb

# Remove parametrized notebooks.
rm test_memory_3.ipynb
rm test_memory_1.ipynb
