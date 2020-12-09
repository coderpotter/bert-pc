# bert-pc
Using BERT and SEG-BERT to solve propositional calculus SAT

# Dependencies
You should install the following in your python environment:
 1. pandas
 2. networkx
 3. simpletransformers
 4. cnfgen
 5. tqdm
 5. PyTorch
 6. transformers

Please see the homepage for each of these packages for installation instructions. For cnfgen, you also need to install a SAT solver it can use to solve for gold labels during the dataset creation. We used cryptominisat:

https://github.com/msoos/cryptominisat

To generate the neuroSAT dataset, we simply used the code released along with the repository:

https://github.com/dselsam/neurosat

Finally, see the following for setting up SEG-BERT:

https://github.com/jwzhanggy/SEG-BERT



# Usage

Each script is used for the following:

 * ```convertDIMACStoCNF.py``` takes as input a SAT dataset in DIMACS format and converts it to S-expressions, which are stored in a pandas dataframe.
 * ```3satTograph.py``` takes as input a SAT dataset in DIMACS format and converts it to graphs which can be used for SegBERT.
 * ```cnfgendataset.py``` generates a purely random SAT dataset.
 * ```graphcnfgenddataset.py``` generates a dataset of graph coloring problems. Since we use this specifically with SEG-BERT it outputs the graph in the format SEG-BERT expects.
 * ```p2.py``` trains and evaluates BERT on the input dataset.
 * ```script_graph_classification.py``` and ```script_preprocess.py``` come from the SEG-BERT project, we have only made minor changes to add support for the graph coloring dataset. After you clone SEG-BERT, you should replace the versions of these files that were cloned with the version from THIS repository. Afterwards, you can delete the versions stored here. You will need to do this for step 3 below to work properly.

Use the following steps to run the code:

 1. For convenience, we provide the satlib dataset along with our code, which can be used to easily train either bert or longformer on sat. Assuming that the satlib folder is in the default location, simply run ```python p2.py```.
 2. To generate the random cnf dataset, you can simply run ```python cnfgendataset.py```. The formulas will be created in a new dataset directory. Then to prepare the input csv, run ```python convertDIMACStoCNF.py```, be sure to change the value of the ```location``` variable towards the top of this file to point to the dataset folder created earlier.
 3. To generate the graph dataset, run ```python graphcnfgendataset.py```, this will create a new dataset under graph-dataset. To train SEG-BERT on this, first run ```python script_preprocess.py```. Make sure that your working directory is the root of this repository when you run it, otherwise the script may not locate the path to the graph-dataset directory, so when you run the previous command you will most likely need to provide the full path to the precprocessing script in the cloned SEG BERT repo. After this finishes, run ```python script_graph_classification.py```. Note that, unlike for bert, the dataset for SEG-BERT is small enough that it can be trained using a cpu.
