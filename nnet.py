from numpy import *
from scipy import *
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.customxml.networkwriter import NetworkWriter
classes = 3
new_data = genfromtxt('NerfGunConradTrainingPhotos/nerfgundata.csv',delimiter=',').transpose()
new_target = genfromtxt('NerfGunConradTrainingPhotos/target.csv',delimiter=',')
numsamples,numvals = new_data.shape #number of datapoints at which an image is analysed, number of images to be analysed
X,y = new_data,new_target#switch to matrix terminology for math portion 
ds = ClassificationDataSet(numvals, 1 , nb_classes=classes)#get the numpy array parsed into a classification dataset
for k in range(len(X)):#iterate over each sample 
    ds.addSample(ravel(X[k]),y[k])#add each sample to the new ClassificationDataSet
tstdatatmp, trndatatmp = ds.splitWithProportion( 0.25 )
tstdata = ClassificationDataSet(numvals, 1, nb_classes=classes)
for n in range(0, tstdatatmp.getLength()):
    tstdata.addSample( tstdatatmp.getSample(n)[0], tstdatatmp.getSample(n)[1] )
trndata = ClassificationDataSet(numvals, 1, nb_classes=classes)
for n in range(0, trndatatmp.getLength()):
    trndata.addSample( trndatatmp.getSample(n)[0], trndatatmp.getSample(n)[1] )
print(type(ds))
trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()
fnn = buildNetwork(trndata.indim, 64 , trndata.outdim, outclass=SoftmaxLayer)#construct a network with random initial values, first value is the dimension of input vectors, second value is dimension of hidden layers, third value is dimension of output
trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1, learningrate=.001 , verbose=True, weightdecay=0.01) 
for i in range (0,200):#run for 5000 epochs
    print('Percent Error on Test dataset after training for epochs: ', i, ' ' , percentError( trainer.testOnClassData (dataset=tstdata ), tstdata['class'] ))#return percent error
    log = open('finallogfile.txt','a')
    log.write(str(percentError(trainer.testOnClassData (dataset=tstdata ), tstdata['class'])))
    log.write('\n')
    log.close()
    trainer.trainEpochs(1)#train one at a time to get percent error at each step
    NetworkWriter.writeToFile(fnn,'nerfgunnetwork.xml')
