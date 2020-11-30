
import tensorflow
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold   
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

def classification_model(model, train_data,train_labels, test_data,test_labels):
  model.fit(train_data,train_labels)
  predictions = model.predict(train_data)
  accuracy = metrics.accuracy_score(predictions,train_labels)
  print('Accuracy : %s' % '{0:.3%}'.format(accuracy))
  test_predictions=model.predict(test_data)
  test_accuracy = metrics.accuracy_score(test_predictions,test_labels)
  print('Test Accuracy : %s' % '{0:.3%}'.format(test_accuracy))

matches=pd.read_csv("matches.csv")

matches['winner'].fillna('Draw',inplace=True)
matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Rising Pune Supergiant','Kochi Tuskers Kerala','Pune Warriors']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','RPS','KTK','PW'],inplace=True)

encode = {'team1': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'team2': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'toss_winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'Draw':14}}
matches.replace(encode, inplace=True)

matches['city'].fillna('Dubai',inplace=True)

x=matches['city']
a=set(x)

cities={}
for i,j in enumerate(a):
  cities[j]=i

matches=matches.replace(cities)

toss={'bat':1,'field':2}
matches=matches.replace(toss)

match_features = matches[['team1','team2','city','toss_decision','toss_winner','winner']]

train, test = train_test_split(match_features, test_size=0.2, random_state=42, shuffle=True)
train_data=train.values
test_data=test.values
train_labels=train_data[:,-1]
train_data=train_data[:,:-1]
test_labels=test_data[:,-1]
test_data=test_data[:,:-1]

model=Sequential([Dense(10,activation='relu',),
                  
                  Dense(128,activation='relu'),
                  Dropout(0.10),
                  Dense(256,activation='relu'),
                  Dropout(0.10),
                  Dense(64,activation='relu'),
                  
                  Dense(15,activation='softmax')])

model.compile(loss='sparse_categorical_crossentropy',optimizer=tensorflow.keras.optimizers.Adam(lr=0.009),metrics=['accuracy'])
model.fit(train_data,train_labels,epochs=400,verbose=1,validation_data=(test_data,test_labels))

var_mod = ['city','toss_decision','venue']
le = LabelEncoder()
match_features1 = matches[['season','team1','team2','city','venue','toss_decision','toss_winner','winner']]
for i in var_mod:
    match_features1[i] = le.fit_transform(match_features1[i])

train, test = train_test_split(match_features1, test_size=0.2, random_state=42, shuffle=True)
train_data=train.values
test_data=test.values
train_labels=train_data[:,-1]
train_data=train_data[:,:-1]
test_labels=test_data[:,-1]
test_data=test_data[:,:-1]

models = [('Naive Bayes', GaussianNB()),
        ('Decision Tree', DecisionTreeClassifier('entropy')),

    
    ('Support Vector Machines', SVC()),
    ('K Nearest Neighbors', KNeighborsClassifier(n_neighbors=8)),
    ('Adaboost ',AdaBoostClassifier()),
    
    ('RandomForesClassifier',RandomForestClassifier()),
    ('Gradient Boosting Classifier',GradientBoostingClassifier(learning_rate=0.03,n_estimators=400))
]

i=0
for name,model in models:
  print
  print("\n",name)
  classification_model(model, train_data,train_labels,test_data,test_labels)
  if(name!='Naive Bayes' and name!='Support Vector Machines' and name!='K Nearest Neighbors'):
    
    
    variables=['season','team1','team2','city','venue','toss_decision','toss_winner']
    imp_input = pd.Series(model.feature_importances_,index=variables).sort_values(ascending=False)
    print("--Importance of each variable in predicting the output--")
    print(imp_input)

