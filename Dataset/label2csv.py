import pandas

path = 'test2_label.txt'
labels = pandas.read_csv(path, sep='\t', header=None,
                        names=['start', 'end', 'annotation'],
                        dtype=dict(start=float,end=float,annotation=str))

print(labels)
print("%.2f" % labels['start'][2])
print(labels['annotation'])

label_onehot = []

print(round(labels['start'][3]*100))
print(round(labels['start'][3]*100)/100)

# 0.01 s per [0] 

time = 8
check = 0

for i in range(0,time*100):
    for j in range(len(labels['start'])):
        if i/0.01 == round(labels['start'][j]*100)/0.01:
            label_onehot.append(1)
            check = 1
    if check == 0:
        label_onehot.append(0)
    check = 0
    
print(label_onehot)