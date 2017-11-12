import pandas as pd
import pickle

def create_dataset():
data=pd.read_csv('testset.csv')
data=data.fillna(0)
print(data.head())
print(data.columns)
condition=data[' _conds']
conds={}
for i in condition:
    try:
        conds[i]+=1
    except KeyError:
        conds[i]=1

condition_index=[]
for k,v in conds.items():
    condition_index.append((v,k))
condition_index=sorted(condition_index,key=lambda x:x[0])
idx=0
conds={}
for i in condition_index:
    print(i[1])
    apparel=[]
    for _ in range(0,5):
        apparel.append(input('Enter an apparel: '))
    conds[idx]=(i[1],apparel)
    idx+=1

print(len(conds))
print(conds)
with open('conds.pickle','wb') as f:
    pickle.dump(conds,f,protocol=pickle.HIGHEST_PROTOCOL)