"""
    Convert examples in DIMACS form to S-expression
    Save the result as a dataframe

    Some of our experiments used S-expressions, in those cases we rand this script first on the input data
    The target directory is expected to contain the DIMACS formula one file at a time.
    Antonio Laverghetta Jr.
    Animesh Nighojkar
"""
import pandas as pd
import re
from os import listdir
from os.path import isfile, join

outfile = pd.DataFrame(columns=["S-exp",'SAT'])
location = 'Z:/Code/neurosat/test/sr5/grp2'

files = [f for f in listdir(location) if isfile(join(location, f))]
getSat = re.compile(r'sat=.')

# get all the files in the directory
# for each dimacs file
    # from the file name, get either 0 or 1 for sat
    # create the list and append AND at the root
    # for each line in the file (skip the header)
        # convert each variable to the sNUMBER, in the appropriate way depending on whether there is a negative
        # OR everything together and append as sublist
    # append to the csv
# save the csv
for f in files:
    s_exp = ['AND']
    label = re.findall(getSat,f)
    if '0' in label[0]:
        label = 0
    else:
        label = 1
    file = open(location + '/' + f,'r')
    for line in file.readlines():
        sub_exp = ['OR']
        # skip the header
        if 'p' in line:
            continue
        line = line.strip().split()
        for string in line:
            if '-' in string:
                sub_exp.append('-s'+string[1:])
            else:
                sub_exp.append('s'+string)
        s_exp.append(sub_exp)
    # clean up the result expression
    s_exp = str(s_exp)
    s_exp = s_exp.replace('[','(').replace(']',')').replace(',','').replace('\'','')
    outfile = outfile.append({'S-exp':s_exp,'SAT':label},ignore_index=True)


outfile.to_csv("./test_final/grp2/test.zip",compression='zip')


