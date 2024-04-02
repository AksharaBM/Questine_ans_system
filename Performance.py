import warnings 
warnings.filterwarnings('ignore') 
from sklearn.random_projection import GaussianRandomProjection
from sklearn import preprocessing 
from keras.models import load_model
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from __pycache__.utils import *
import matplotlib.font_manager as font_manager
from numpy import asarray, mean
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import keras
from transformers import TFBertModel
import tensorflow as tf

def Test_pred():
    test_data=np.load("files/x_test.npy")
    test_label=np.load("files/y_test.npy")
    with keras.utils.custom_object_scope({'TFBertModel': TFBertModel}):
        model = load_model("Model/RBTM.h5")
    general_similarity = tf.constant([[test_data]])
    domain_similarity = tf.constant([[test_label]])
    num_additional_features = 10  
    additional_input_example = tf.random.uniform((1, num_additional_features))
    logits = model([general_similarity, domain_similarity, additional_input_example])
    
    cnn_lstm_model=load_model("Model/CNN_LSTM.h5")
    cnn_lstm_prediction = cnn_lstm_model.predict([test_data, test_label])

    bilstm_model=load_model("Model/BiLSTM.h5")
    bilstm_prediction = bilstm_model.predict([test_data, test_label])
    
    autoencoder_model=load_model("Model/autoencoder.h5")
    autoencode_prediction = autoencoder_model.predict([test_data, test_label])
    



        
def Plot():
    Class= np.unique(np.load("files/y_test.npy"))
    X=len(np.unique(Class))
    X=predict(X)
    y_test=X[:,0]  ;pred=X[:,1] 
    
    
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)

    # Accuracy=1-mae
    def calculate_relative_error(true, pred):
        return abs(true - pred) / true

    def calculate_accuracy_relative(y_true, y_pred, threshold):
        correct_predictions = 0
        total_predictions = len(y_true)
    
        for true, pred in zip(y_true, y_pred):
            relative_error = calculate_relative_error(true, pred)
            if relative_error <= threshold:
                correct_predictions += 1
    
        accuracy = (correct_predictions / total_predictions) 
        return accuracy


    threshold = 0.05  
    accuracy = calculate_accuracy_relative(y_test,pred, threshold)
  
    
    print("proposed performance : \n********************\n")
    print("Accuracy            :",accuracy)
    print("R2-Score            :",r2)
    print('MAE                 :',mae)
    print("MSE                 :",mse) 
    print("RMSE                :",rmse)
    print()
 

    #--------------------------------#
    # -------------Auto Encoder------------#
    pred_cl=X[:,2] 
    mae_cl = mean_absolute_error(y_test, pred_cl)
    mse_cl = mean_squared_error(y_test, pred_cl)
    rmse_cl = np.sqrt(mse_cl)
    r2_cl = r2_score(y_test, pred_cl)

    
    def calculate_relative_error(true, pred):
        return abs(true - pred) / true

    def calculate_accuracy_relative(y_true, y_pred, threshold):
        correct_predictions = 0
        total_predictions = len(y_true)
    
        for true, pred in zip(y_true, y_pred):
            relative_error = calculate_relative_error(true, pred)
            if relative_error <= threshold:
                correct_predictions += 1
    
        accuracy = (correct_predictions / total_predictions) 
        return accuracy


    threshold = 0.05  
    accuracy_cl = calculate_accuracy_relative(y_test,pred_cl, threshold)
  
 
    print("Auto Encoder: \n********************\n")
    print("Accuracy            :",accuracy_cl)
    print("R2-Score            :",r2_cl)
    print('MAE                 :',mae_cl)
    print("MSE                 :",mse_cl) 
    print("RMSE                :",rmse_cl)
 
    

    # -------------CNN-LSTM------------#
    pred_bi=X[:,3] 
    mae_bi = mean_absolute_error(y_test, pred_bi)
    mse_bi = mean_squared_error(y_test, pred_bi)
    rmse_bi = np.sqrt(mse_bi)
    r2_bi = r2_score(y_test, pred_bi)
    
   
    def calculate_relative_error(true, pred):
        return abs(true - pred) / true
    
    def calculate_accuracy_relative(y_true, y_pred, threshold):
        correct_predictions = 0
        total_predictions = len(y_true)
        
        for true, pred in zip(y_true, y_pred):
            relative_error = calculate_relative_error(true, pred)
            if relative_error <= threshold:
                correct_predictions += 1
    
        accuracy = (correct_predictions / total_predictions) 
        return accuracy
    
    
    threshold = 0.05  
    accuracy_bi = calculate_accuracy_relative(y_test,pred_bi, threshold)

    print()
    print("CNN-LSTM performance: \n********************\n")
    print("Accuracy            :",accuracy_bi)
    print("R2-Score            :",r2_bi)
    print('MAE                 :',mae_bi)
    print("MSE                 :",mse_bi) 
    print("RMSE                :",rmse_bi)
    
    
    
    #------------------Bi-LSTM----------------
    pred_ae=X[:,4] 
    mae_ae = mean_absolute_error(y_test, pred_ae)
    mse_ae = mean_squared_error(y_test, pred_ae)
    rmse_ae = np.sqrt(mse_ae)
    r2_ae = r2_score(y_test, pred_ae)
    
    def calculate_relative_error(true, pred):
        return abs(true - pred) / true
    
    def calculate_accuracy_relative(y_true, y_pred, threshold):
        correct_predictions = 0
        total_predictions = len(y_true)
        
        for true, pred in zip(y_true, y_pred):
            relative_error = calculate_relative_error(true, pred)
            if relative_error <= threshold:
                correct_predictions += 1
    
        accuracy = (correct_predictions / total_predictions) 
        return accuracy
    
    
    threshold = 0.05 
    accuracy_ae = calculate_accuracy_relative(y_test,pred_ae, threshold)

    print()
    print("Bi-LSTM performance: \n********************\n")
    print("Accuracy            :",accuracy_ae)
    print("R2-Score            :",r2_ae)
    print('MAE                 :',mae_ae)
    print("MSE                 :",mse_ae) 
    print("RMSE                :",rmse_ae)
    
    
    legend_properties = {'weight': 'bold', 'family': 'Times New Roman', 'size': 14}
    con = "Proposed"
    con1 = "  AE" 
    con2 = "    CNN-LSTM"
    con3 = "Bi-LSTM"
    con4="BERT"
     
    
    plt.figure(figsize=(7,5));plt.ylim(85,100)    
    width = 0.25
    plt.bar(0,accuracy*100, width, color='#1E90FF', align='center', edgecolor='black',) 
    plt.bar(1,accuracy_cl*100, width, color='#40E0D0', align='center', edgecolor='black') 
    plt.bar(2,accuracy_bi*100, width, color='#DB7093', align='center', edgecolor='black') 
    plt.bar(3,accuracy_ae*100, width, color='#808080', align='center', edgecolor='black') 
   
    
    
    plt.xticks(np.arange(4),(con, con1,con2,con3),fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('Accuracy (%)',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Results/Accuracy.png', format="png",dpi=600)


    plt.figure(figsize=(7,5));plt.ylim(85,100)    
    width = 0.25
    plt.bar(0,r2*100, width, color='#1E90FF', align='center', edgecolor='black',) 
    plt.bar(1,r2_cl*100, width, color='#40E0D0', align='center', edgecolor='black') 
    plt.bar(2,r2_bi*100, width, color='#DB7093', align='center', edgecolor='black') 
    plt.bar(3,r2_ae*100, width, color='#808080', align='center', edgecolor='black') 
 
    

    plt.xticks(np.arange(4),(con, con1,con2,con3),fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('R2-Score (%)',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Results/r2-score.png', format="png",dpi=600)
   

    plt.figure(figsize=(7.2,5));plt.ylim(0,0.025)
    width = 0.25
    plt.bar(0,mae, width, color='#1E90FF', align='center', edgecolor='black',) 
    plt.bar(1,mae_cl, width, color='#40E0D0', align='center', edgecolor='black') 
    plt.bar(2,mae_bi, width, color='#DB7093', align='center', edgecolor='black') 
    plt.bar(3,mae_ae, width, color='#808080', align='center', edgecolor='black') 
    
    

    plt.xticks(np.arange(4),(con, con1,con2,con3),fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('MAE',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Results/MAE.png', format="png",dpi=600) 
    
    plt.figure(figsize=(7.2,5))
    width = 0.25
    plt.bar(0,mse, width, color='#1E90FF', align='center', edgecolor='black',) 
    plt.bar(1,mse_cl, width, color='#40E0D0', align='center', edgecolor='black') 
    plt.bar(2,mse_bi, width, color='#DB7093', align='center', edgecolor='black') 
    plt.bar(3,mse_ae, width, color='#808080', align='center', edgecolor='black') 
  
    

    plt.xticks(np.arange(4),(con, con1,con2,con3),fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('MSE',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Results/MSE.png', format="png",dpi=600)
    
    plt.figure(figsize=(7,5))
    width = 0.25
    plt.bar(0,rmse, width, color='#1E90FF', align='center', edgecolor='black',) 
    plt.bar(1,rmse_cl, width, color='#40E0D0', align='center', edgecolor='black') 
    plt.bar(2,rmse_bi, width, color='#DB7093', align='center', edgecolor='black') 
    plt.bar(3,rmse_ae, width, color='#808080', align='center', edgecolor='black') 
   
    

    plt.xticks(np.arange(4),(con, con1,con2,con3),fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('RMSE',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Results/RMSE.png', format="png",dpi=600)

def predict(X):
    pred=Predict(X)
    return pred 
  