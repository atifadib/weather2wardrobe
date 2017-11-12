import requests
from pprint import pprint
import json


def getWeather(location):
    req=requests.get('https://query.yahooapis.com/v1/public/yql?q=select%20*%20from%20weather.forecast%20where%20woeid%20in%20(select%20woeid%20from%20geo.places(1)%20where%20text%3D%22'+location+'%22)&format=json&diagnostics=true&env=store%3A%2F%2Fdatatables.org%2Falltableswithkeys&callback=')

    js=json.loads(req.text)
    img=js['query']['results']['channel']['item']['condition']['code']
    url='http://l.yimg.com/a/i/us/we/52/'+img+'.gif'
    hum=js['query']['results']['channel']['atmosphere']['humidity']
    pres=js['query']['results']['channel']['atmosphere']['pressure']
    visi=js['query']['results']['channel']['atmosphere']['visibility']
    temp=js['query']['results']['channel']['item']['condition']['temp']
    wdir=js['query']['results']['channel']['wind']['direction']
    wspd=js['query']['results']['channel']['wind']['speed']
    forecast=js['query']['results']['channel']['item']['forecast'][0]['code']
    hum=float(hum)
    pres=float(pres)
    temp=float(temp)
    visi=float(visi)
    wdir=float(wdir)
    wspd=float(wspd)
    return [hum/250,pres/101061443.0,temp/100,visi/6000,wdir/1000,wspd/1500]#,forecast)
