#!/usr/bin/env python
# coding: utf-8

# # Model Usage

# In[1]:


def model_usage():
    # Importing joblib to load the model file
    from joblib import load,dump
    import warnings
    warnings.filterwarnings('ignore')
    
    #Loading the file
    model = load("Wine_Quality.joblib")

    import numpy as np
    
    #Predicting the wine quality given the features 
    features = np.array([[ 8.     ,  0.38   ,  0.06   ,  1.8    ,  0.078  , 12.     ,
           49.     ,  0.99625,  3.37   ,  0.52   ,  9.9    ]])
    predictions = model.predict(features)
    print("Features:",features)
    print("Prediction:",predictions)


# ## As we can see, given the features, model can predict the given value
