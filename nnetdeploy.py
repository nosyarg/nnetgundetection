import urllib.request
from numpy import *
from scipy import *
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
fnn = NetworkReader.readFrom('nerfgunnetwork.xml')
print("NETWORK LOADED")
classes = 3
new_data = genfromtxt('NerfGunConradTrainingPhotos/tempimages/nerfgundata.csv',delimiter=',').transpose()
print("DATA LOADED")
numsamples,numvals = new_data.shape #number of datapoints at which an image is analysed, number of images to be analysed
X,y = new_data,1#switch to matrix terminology for math portion 
for i in range(len(X)):
        ds = ClassificationDataSet(numvals, 1 , nb_classes=classes)#get the numpy array parsed into a classification dataset
        ds.addSample(ravel(X[i]),1)#add each sample to the new ClassificationDataSet
        ds._convertToOneOfMany()
        trainer = BackpropTrainer(fnn)
        print( percentError( trainer.testOnClassData (dataset=ds ), ds['class'] ))#return percent error
        print(i)
        if(0):# percentError( trainer.testOnClassData (dataset=ds ), ds['class'] )>50):#return percent error
                urllib.request.urlopen("https://rest.nexmo.com/sms/json?api_key=3ce4c35a&api_secret=616e4a499ce72ce1&from=12134657970&to=13369712976&text=A+weapon+has+been+detected+by+camera+on+Hunt+East+4th+floor")
                exit()
