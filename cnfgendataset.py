"""
    Used to generate the purely random cnf dataset
    Antonio Laverghetta Jr.
    Animesh Nighojkar
"""

import cnfgen

for i in range(0,100001):
    formula = cnfgen.families.randomformulas.RandomKCNF(10,10,20)
    sat = formula.is_satisfiable()[0]
    file = open(f'./dataset/formula{i}_{sat}.dimacs','w+')
    file.writelines(formula.dimacs())
    file.close()