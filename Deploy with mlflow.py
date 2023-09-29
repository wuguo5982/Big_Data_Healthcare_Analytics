"""
Apply mlFlow for machine learning and deeep learning models.
"""

!pip install mlflow
!pip install pyngrok

get_ipython().system_raw("mlflow ui --port 5000 &")

from pyngrok import ngrok

ngrok.kill()

ngrok_tunnel = ngrok.connect(addr="5000", proto="http", bind_tls=True)

print("MLflow UI ", ngrok_tunnel.public_url)

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Sample 1
if __name__ == "__main__":

    mlflow.set_experiment(experiment_name="mlflow test001")
    
    training_data = pd.read_csv('../electronic_medical_record001.csv')
    print("loaded training data")

    training_data.describe()
    mlflow.log_param("training percentage",50)

    mlflow.log_param("dataset shape",training_data.shape)
    
    X = training_data.iloc[:, :-1].values
    y = training_data.iloc[:,-1].values
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.2,random_state=25)    

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    print("Feature Scaling")
    
    
    # Minkowski (p=2): euclidean distance)
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    
    # Model training
    classifier.fit(X_train, y_train)
    print("Model training")
   
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:,1]
    
        
    cm = confusion_matrix(y_test, y_pred)  
    
    model_accuracy = accuracy_score(y_test,y_pred)
    
    print(model_accuracy)
    
    mlflow.log_metric("accuracy", model_accuracy)
    mlflow.sklearn.log_model(classifier, "model")

######################################################
# Sample 2
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch

if __name__ == "__main__":
    
    mlflow.set_experiment(experiment_name="mlflow pytorch test002")    
    dataset = pd.read_csv('../electronic_medical_record002.csv')
    
    dataset.describe()    
    dataset.head()
    
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:,-1].values
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.20,random_state=0)
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    Xtrain = torch.from_numpy(X_train).float()
    Xtest = torch.from_numpy(X_test).float()
    
    Xtrain
    
    ytrain = torch.from_numpy(y_train)
    ytest = torch.from_numpy(y_test)
    
    ytrain
    
    Xtrain.shape, ytrain_.shape, Xtest.shape, ytest.shape
    
    input_size=2
    output_size=2
    hidden_size=10
    
    class Net(nn.Module):
       def __init__(self):
           super(Net, self).__init__()
           self.fc1 = torch.nn.Linear(input_size, hidden_size)
           self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
           self.fc3 = torch.nn.Linear(hidden_size, output_size)    
    
       def forward(self, X):
           X = torch.relu((self.fc1(X)))
           X = torch.relu((self.fc2(X)))
           X = self.fc3(X)
    
           return F.log_softmax(X,dim=1)
    
    model = Net()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.NLLLoss()
    
    epochs = 100
    
    for epoch in range(epochs):
      optimizer.zero_grad()
      Ypred = model(Xtrain)
      loss = loss_fn(Ypred,  ytrain)
      loss.backward()
      optimizer.step()
      print('Epoch',epoch, 'loss',loss.item())
    
    mlflow.end_run()
    with mlflow.start_run() as run:
        mlflow.log_param("epochs", 50)
        mlflow.pytorch.log_model(model, "models")
        mlflow.log_metric("loss", loss.item())

