from flask import Flask,render_template,request
from getWeather import getWeather
import pickle
import numpy as np
from random import randint

global svm,tree,condition
with open('conds.pickle','rb') as f:
    conditions=pickle.load(f)
with open('models.pickle','rb') as f:
    models=pickle.load(f)
svm=models[0]
tree=models[1]

app=Flask(__name__,template_folder=r'.\template')

@app.route('/',methods=['POST','GET'])
def index():
    location="Bangalore,Karnataka"
    if(request.method=='POST'):
        location=request.form['location']
    output=getWeather(location)
    sample=np.array(output).reshape(-1,6)
    out1=svm.predict(sample)
    out2=svm.predict(sample)
    out=round((out1[0]+out2[0])/2)
    out=int(out)
    print(out)
    out=randint(0,40)
    ico='http://l.yimg.com/a/i/us/we/52/'+str(out)+'.gif'
    data=conditions[out]
    print(data)
    print(out)
    print(output)
    return render_template('index.html',output=output,data=data,ico=ico,location=location)

if(__name__=="__main__"):
    app.run(debug=True)
