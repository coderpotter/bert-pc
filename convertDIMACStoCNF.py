"""
    Convert examples in DIMACS form to S-expression
    Save the result as a dataframe
"""
import pandas as pd
import re
from os import listdir
from os.path import isfile, join

# outfile = pd.DataFrame(columns=["S-exp",'SAT'])
outfile = pd.DataFrame(columns=["text",'labels'])
# location = 'Z:/Code/neurosat/test/sr5/grp2'
location = 'satlib/uuf50-218'

files = [f for f in listdir(location) if isfile(join(location, f))]
# getSat = re.compile(r'sat=.')

for f in files:
    s_exp = ['AND']
    # label = re.findall(getSat,f)
    # if '0' in label[0]:
    #     label = 0
    # else:
    #     label = 1
    if '/uf' in location:
        label = 1
    else:
        label = 0
    # file = open(location + '/' + f,'r')
    file = open(location + '/' + f)
    for line in file.readlines():
        sub_exp = ['OR']
        # skip the header
        if 'p' in line or 'c' in line:
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
    outfile = outfile.append({'text':s_exp,'labels':label},ignore_index=True)


# outfile.to_csv("./test_final/grp2/test.zip",compression='zip')
outfile.to_csv(location + '.csv')

# get all the files in the directory
# for each dimacs file
    # from the file name, get either 0 or 1 for sat
    # create the list and append AND at the root
    # for each line in the file (skip the header)
        # convert each variable to the sNUMBER, in the appropriate way depending on whether there is a negative
        # OR everything together and append as sublist
    # append to the csv
# save the csv

