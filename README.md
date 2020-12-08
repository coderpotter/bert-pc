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

 * ```convertDIMACStoCNF.py``` takes input a SAT dataset in DIMACS form converts it to S-expressions, in a pandas dataframe
 * ```cnfgendataset.py``` generates a purely random SAT dataset.
 * ```graphcnfgenddataset.py``` generates a dataset of graph coloring problems. Since we use this specifically with SEG-BERT it outputs the graph in the format SEG-BERT expects.
 * ```p2.py``` trains and evaluates BERT on the input dataset.

After setting up the dependencies, follow these steps to run the code.

 1. 
