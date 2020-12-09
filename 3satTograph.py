'''
Antonio Laverghetta Jr., Animesh Nighojkar
Takes as input a SAT dataset in DIMACS format and converts it to graphs which can be used for SegBERT.
'''

op = open('segbert/graph-dataset/satlib-graphs/data.txt', 'w+')
op.writelines('200\n')

dataset = '/raid/home/animesh/storage/bert-pc/satlib/uf250-1065/uf250-0'
for i in range(1, 101):
    cnf = dataset+str(i)+'.cnf'
    with open(cnf, 'r') as f:
        for l in f.readlines():
            if l[0] == 'c':
                continue
            l = l.strip().split()
            if not l:
                continue
            if l[0] == 'p':
                n_literals, n_clauses = int(l[2]), int(l[3])
                op.writelines(str(n_clauses + 2*(n_literals)) + ' 1\n')
                continue
            l = list(map(int, l[:-1]))
            if not l:
                continue
            line = [0]
            for _ in l:
                if _ < 0:
                    line.append(n_literals + abs(_))
                else:
                    line.append(_)
            op.writelines(' '.join(list(map(str, line)))+'\n')
        for _ in range(1, n_literals+1):
            op.writelines('0 '+str(_)+'\n')
        for _ in range(1, n_literals+1):
            op.writelines('0 '+str(n_literals+_)+'\n')
dataset = '/raid/home/animesh/storage/bert-pc/satlib/uuf250-1065/uuf250-0'
for i in range(1, 101):
    cnf = dataset+str(i)+'.cnf'
    with open(cnf, 'r') as f:
        for l in f.readlines():
            if l[0] == 'c':
                continue
            l = l.strip().split()
            if not l:
                continue
            if l[0] == 'p':
                n_literals, n_clauses = int(l[2]), int(l[3])
                op.writelines(str(n_clauses + 2*(n_literals)) + ' 0\n')
                continue
            l = list(map(int, l[:-1]))
            if not l:
                continue
            line = [0]
            for _ in l:
                if _ < 0:
                    line.append(n_literals + abs(_))
                else:
                    line.append(_)
            op.writelines(' '.join(list(map(str, line)))+'\n')
        for _ in range(1, n_literals+1):
            op.writelines('0 '+str(_)+'\n')
        for _ in range(1, n_literals+1):
            op.writelines('0 '+str(n_literals+_)+'\n')
