from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm 
from os import listdir

directory = "../Project_Pattern/classification_dataset/"#change the directory
dic = open("../Project_Pattern/labels.txt",'r')
dic_labels = ast.literal_eval(dic.read())

def load_photos_and_construct_feature_vector_and_add_labels_list(directory,dic):
    labels=[]
    pixels=[]
    for name in listdir(directory):
        filename = directory+'/'+name
        img =load_img(filename,target_size=(224,224))
        #convert img to numpy array
        img_array =img_to_array(img)
        img_array = img_array.flatten()
        if dic[name.split('.jpg')[0]] == 'dog':
            labels.append(1)
        else: 
            labels.append(0)
        pixels.append(img_array)
    return pixels,labels 
 
model_2 = svm.SVC()
inp,out=load_photos_and_construct_feature_vector_and_add_labels_list(directory,dic_labels)
#print(out)
model_1 = KNeighborsClassifier()
model_2 = svm.SVC()

xTrain,xTest,yTrain,yTest=train_test_split(inp,out,test_size=0.2,random_state=5)
def contstruct_model_and_eval(model,xTrain,yTrain,xTest,yTest):
    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)
    yTrain=yTrain.reshape(yTrain.shape[0],1)
    xTest = np.array(xTest)
    model.fit(xTrain,yTrain)
    y_pred=model.predict(xTest)
    y_pred=y_pred.reshape(y_pred.shape[0],1)
    yTest = np.array(yTest)
    yTest = yTest.reshape(yTest.shape[0],1)
    j=0
    for i in range(len(y_pred)):
        sub = y_pred[i]-yTest[i]
        if sub == 0 :
            j=j+1
    acc=(j/y_pred.shape[0])*100
    print("Accuracy ",acc,"%")
 
contstruct_model_and_eval(model_1,xTrain,yTrain,xTest,yTest)

xTrain,xTest,yTrain,yTest=train_test_split(inp,out,test_size=0.2,random_state=100)
def contstruct_model_and_eval(model,xTrain,yTrain,xTest,yTest):
    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)
    yTrain=yTrain.reshape(yTrain.shape[0],1)
    xTest = np.array(xTest)
    model.fit(xTrain,yTrain)
    y_pred=model.predict(xTest)
    y_pred=y_pred.reshape(y_pred.shape[0],1)
    yTest = np.array(yTest)
    yTest = yTest.reshape(yTest.shape[0],1)
    j=0
    for i in range(len(y_pred)):
        sub = y_pred[i]-yTest[i]
        if sub == 0 :
            j=j+1
    acc=(j/y_pred.shape[0])*100
    print("Accuracy ",acc,"%")
contstruct_model_and_eval(model_2,xTrain,yTrain,xTest,yTest)
