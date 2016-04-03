#Grayson York, Adithya Balaji, Kireet Panuganti
#Neural network for firearm detection image parsing
setwd("/Users/grayson/Desktop/Conrad Files/trainingdata")#input directory of training images here
#source("http://bioconductor.org/biocLite.R")
#biocLite("EBImage")
#uncomment above two lines to install the EBImage library
library(EBImage)
#Edit these to change the factor by which the image is scaled down in each direction
scalefactorhoriz <- 64
scalefactorvert <- 48
#End editable section
#dir.create("images")#make a directory to hold the renamed images
names <- Sys.glob("*.JPG") #take all of the images in the training set and load their filenames to names
#for(i in 1:length(names))#rename all images to 1,2,3,4... to facilitate negative mining
#{
#  imagetmp <- readImage(names[i])#read the current image to our memory
#  writeImage(imagetmp,paste("images/",i,".jpg",sep='',collapse=''))#write the renamed images to the new files
#}
color.image <- readImage(names[1]) #read the first image found into memory
bw.image <- channel(color.image,"gray")#convert the image from RGB to black and white
imgsize <- dim(bw.image) #get the number of pixels in each direction
datasize <- imgsize[1]*imgsize[2]/(scalefactorvert*scalefactorhoriz)#get the size which our vector will be once it is completely scaled down
bw.image <- transpose(matrix(as.array(bw.image[seq(1,imgsize[1]*imgsize[2],scalefactorhoriz)]),imgsize[1]/scalefactorhoriz))#scale the image down horizontally
bw.image <- transpose(matrix(as.array(bw.image[seq(1,imgsize[2]*imgsize[1]/scalefactorhoriz,scalefactorvert)]),imgsize[2]/scalefactorvert))#scale the image down vertically
data <- as.vector(bw.image)#unroll the matrix into one vector so that we can add it to a csv in a useful form
for(i in 2:length(names))#loop the previous steps over the remaining images in the directory
{
print(i)
color.image <- readImage(names[i])#read in the image in full color
bw.image <- channel(color.image,"gray")#grayscale image
bw.image <- transpose(matrix(as.array(bw.image[seq(1,imgsize[1]*imgsize[2],scalefactorhoriz)]),imgsize[1]/scalefactorhoriz))#scale horizontal
bw.image <- transpose(matrix(as.array(bw.image[seq(1,imgsize[2]*imgsize[1]/scalefactorhoriz,scalefactorvert)]),imgsize[2]/scalefactorvert))#scalevertical
data <- c(data,as.vector(bw.image)) #concatenate the data from the new image onto our dataset
}
setwd('..')#write outside the training data
write.table(matrix(data,datasize),file="compresseddata.csv",row.names = FALSE, col.names = FALSE,sep=",")#write compiled dataset to a csv file
