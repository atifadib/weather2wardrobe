import pandas as pd
import pickle
from sklearn import svm
from sklearn import tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def create_dataset():
    with open('conds.pickle','rb') as f:
        conds=pickle.load(f)
    conditions={}
    for k,v in conds.items():
        conditions[v[0]]=(k,v[1])

    data=pd.read_csv('testset.csv')
    data=shuffle(data)
    train,test=train_test_split(data,test_size=0.05,random_state=42)
    test=test.fillna(0)
    train=train.fillna(0)
    features=[
    #' _dewptm',
    #' _fog',
    #' _hail',
    #' _heatindexm',
    ' _hum',
    ' _pressurem',
    #' _rain',
    ' _tempm',
    #' _thunder',
    ' _vism',
    ' _wdird',
    ' _wspdm',
    ]
    for i in features:
        test[i]=test[i]/max(data[i])
        train[i]=train[i]/max(data[i])
        print(i,max(data[i]))
    test=test.fillna(0)
    train=train.fillna(0)

    train_X,train_Y=train[features],train[' _conds'].apply(lambda x:conditions[x][0])
    new_train_Y=[]
    for i in train_Y.values:
        sample=[0 for i in range(40)]
        sample[i]+=1
        new_train_Y.append(sample)
    ret_train_Y=train_Y
    train_Y=new_train_Y
    # print(train_Y.head())
    # print(train_X.head())
    test_X,test_Y=test[features],test[' _conds'].apply(lambda x:conditions[x][0])
    new_test_Y=[]
    for i in test_Y.values:
        sample=[0 for i in range(40)]
        sample[i]+=1
        new_test_Y.append(sample)
    ret_test_Y=test_Y
    test_Y=new_test_Y
    return train_X.values,train_Y,ret_train_Y,test_X.values,test_Y,ret_test_Y

if(__name__=='__main__'):
    train_X,train_Y,ret_train_Y,test_X,test_Y,ret_test_Y=create_dataset()
    print('Training an SVM.....')
    clf_svm=svm.SVC(kernel='rbf')
    clf_svm=clf_svm.fit(train_X,ret_train_Y)
    clf_tree=tree.DecisionTreeClassifier()
    clf_tree=clf_tree.fit(train_X,ret_train_Y)
    with open('models.pickle','wb') as f:
       pickle.dump([clf_svm,clf_tree],f,protocol=pickle.HIGHEST_PROTOCOL)
    with open('models.pickle','rb') as f:
       clfs=pickle.load(f)
    clf_svm=clfs[0]
    clf_tree=clfs[1]
    correct_svm,correct_tree,total=0,0,0
    print('testing SVM')
    print('testing on: ',len(test_X))
    for i,j in zip(test_X,ret_test_Y):
        output_class_svm=clf_svm.predict(i.reshape(-1,6))
        output_class_tree=clf_tree.predict(i.reshape(-1,6))
        output_class_svm=list(output_class_svm)
        output_class_tree=list(output_class_tree)
        #print(output_class,j)
        if(output_class_svm[0]==j):
            correct_svm+=1
        if(output_class_tree[0]==j):
            correct_tree+=1
        total+=1
    print('Accuracy-SVM: ', correct_svm/total)
    print('Accuracy-Tree: ', correct_tree/total)
