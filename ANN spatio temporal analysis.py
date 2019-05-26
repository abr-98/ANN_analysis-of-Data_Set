#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df_k=pd.read_csv('6mar.csv')
df=df_k[df_k.Timelevel==1]

model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.2,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
        if y_test_2.iloc[j][3]==1:
            y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)
    
    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])
    
    else:
    
        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1
    
#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)


# Accuracy Timelevel 1
# 

# In[3]:


import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df_k=pd.read_csv('6mar.csv')
df=df_k[df_k.Timelevel==2]

model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.2,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
        if y_test_2.iloc[j][3]==1:
            y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)
    
    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])
    
    else:
    
        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1
    
#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)


# accuracy Timelevel 2

# In[4]:


import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df_k=pd.read_csv('6mar.csv')
df=df_k[df_k.Timelevel==3]

model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.2,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
        if y_test_2.iloc[j][3]==1:
            y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)
    
    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])
    
    else:
    
        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1
    
#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)


# In[ ]:





# accuracy Timelevel 3

# In[6]:


import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df_k=pd.read_csv('6mar.csv')
df=df_k[df_k.Timelevel==4]

model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.2,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
        if y_test_2.iloc[j][3]==1:
            y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)
    
    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])
    
    else:
    
        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1
    
#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)


# accuracy timelevel 4

# In[7]:


import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df_k=pd.read_csv('6mar.csv')
df=df_k[df_k.Zone=='highway']

model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.2,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
        if y_test_2.iloc[j][3]==1:
            y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)
    
    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])
    
    else:
    
        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1
    
#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)


# accuracy zone: highway

# In[8]:


import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df_k=pd.read_csv('6mar.csv')
df=df_k[df_k.Zone=='market']

model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.2,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
        if y_test_2.iloc[j][3]==1:
            y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)
    
    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])
    
    else:
    
        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1
    
#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)


# accuracy zone:market

# In[9]:


import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df_k=pd.read_csv('6mar.csv')
df=df_k[df_k.Zone=='normal_city']

model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.2,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
        if y_test_2.iloc[j][3]==1:
            y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)
    
    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])
    
    else:
    
        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1
    
#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)


# accuracy zone:normal_city

# In[12]:


import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df_k=pd.read_csv('6mar.csv')
df=df_k[(df_k.Zone=='normal_city') & (df_k.Timelevel==1)]

model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.2,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
        if y_test_2.iloc[j][3]==1:
            y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)
    
    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])
    
    else:
    
        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1
    
#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)


# accuracy: normal_city and 1

# In[14]:


import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df_k=pd.read_csv('6mar.csv')
df=df_k[(df_k.Zone=='normal_city') & (df_k.Timelevel==2)]

model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.2,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
        if y_test_2.iloc[j][3]==1:
            y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)
    
    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])
    
    else:
    
        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1
    
#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)


# accuracy:normal_city and 2

# In[21]:


import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df_k=pd.read_csv('6mar.csv')
df=df_k[(df_k.Zone=='normal_city') & (df_k.Timelevel==3)]

model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.2,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
#scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
      #  if y_test_2.iloc[j][3]==1:
            #y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)
    
    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])
    
    else:
    
        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1
    
#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)


# normal_city and 3

# Very_fast absent

# In[22]:


import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df_k=pd.read_csv('6mar.csv')
df=df_k[(df_k.Zone=='normal_city') & (df_k.Timelevel==4)]

model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.2,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
        if y_test_2.iloc[j][3]==1:
            y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)
    
    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])
    
    else:
    
        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1
    
#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)


# accuracy normal_city and 4

# In[2]:


import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df_k=pd.read_csv('6mar.csv')
df=df_k[(df_k.Zone=='highway') & (df_k.Timelevel==1)]

model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.2,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
#scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
       # if y_test_2.iloc[j][3]==1:
        #    y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)
    
    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])
    
    else:
    
        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1
    
#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)


# Accuracy: highway timelevel 1

# In[4]:


import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df_k=pd.read_csv('6mar.csv')
df=df_k[(df_k.Zone=='highway') & (df_k.Timelevel==2)]

model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.2,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
        if y_test_2.iloc[j][3]==1:
            y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)
    
    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])
    
    else:
    
        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1
    
#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)


# accuracy zone: highway and 2

# In[22]:


import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df_k=pd.read_csv('6mar.csv')
df=df_k[(df_k.Zone=='highway') & (df_k.Timelevel==3)]

model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.2,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
#scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
        if y_test_2.iloc[j][3]==1:
            y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)
    
    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])
    
    else:
    
        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1
    
#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)


# Accuracy:highway and 3

# In[6]:


import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df_k=pd.read_csv('6mar.csv')
df=df_k[(df_k.Zone=='highway') & (df_k.Timelevel==4)]

model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.2,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
        if y_test_2.iloc[j][3]==1:
            y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)
    
    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])
    
    else:
    
        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1
    
#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)


# Accuracy: highway and 4

# In[7]:


import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df_k=pd.read_csv('6mar.csv')
df=df_k[(df_k.Zone=='market') & (df_k.Timelevel==1)]

model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.2,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
        if y_test_2.iloc[j][3]==1:
            y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)
    
    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])
    
    else:
    
        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1
    
#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)


# Accuracy: market and 1

# In[20]:


import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df_k=pd.read_csv('6mar.csv')
df=df_k[(df_k.Zone=='market') & (df_k.Timelevel==2)]

model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.2,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
#scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
        #if y_test_2.iloc[j][3]==1:
         #   y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)
    
    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])
    
    else:
    
        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1
    
#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)


# Accuracy: market and 2

# In[23]:


import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df_k=pd.read_csv('6mar.csv')
df=df_k[(df_k.Zone=='market') & (df_k.Timelevel==3)]

model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.2,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
#scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
        if y_test_2.iloc[j][3]==1:
            y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)
    
    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])
    
    else:
    
        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1
    
#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)


# Accuracy:market and 3

# In[13]:


import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df_k=pd.read_csv('6mar.csv')
df=df_k[(df_k.Zone=='market') & (df_k.Timelevel==4)]

model=Sequential()
#print(len(df))

#TRain
X=df[['Honk_duration','Road_surface','Intersection density','WiFi density','Timelevel']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_train, X_test, y_train, y_test_k = train_test_split(X_d,y_d,test_size=0.2,random_state=42)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test_k[0].copy()

#value_check=new_y.tolist()
#print(value_check)

y_test=pd.DataFrame(new_y)

y_train_2=pd.get_dummies(new_y_d_2)
y_test_2=pd.get_dummies(new_y)
n_cols=X_train.shape[1]
print(n_cols)

model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1200, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=3)
#X_d_2=to_categorical(X_d)
#y_d_2=to_categorical(y_d)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_2, epochs=160, callbacks=[early_stopping_monitor],batch_size=50)


#3
# evaluate the model
#scores = model.evaluate(X_test, y_test_2)
scores_2 = model.evaluate(X_train, y_train_2)
#print(X_test)
#print(y_test)
#new_y_2=y_train[0].copy()
#new_y_d_2=pd.DataFrame(new_y_2)
#new_y=y_test[0].copy()
predictions=model.predict(X_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
#value_check=new_y.tolist()
#print(value_check)
#print(value_check)

#---------------------#
#new_y_d=pd.DataFrame(new_y)
#----------------------#
speed_check=y_test_k[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #
y_test_l=[]

l=len(y_test_2)
j=0
while j<l:
   # if j not in all_zero:
        if y_test_2.iloc[j][0]==1:
            y_test_l.append('Fast')
        if y_test_2.iloc[j][1]==1:
            y_test_l.append('Normal')
        if y_test_2.iloc[j][2]==1:
            y_test_l.append('Slow')
        if y_test_2.iloc[j][3]==1:
            y_test_l.append('Very Fast')
        j=j+1
prediction_l=[]
count=0
all_zero=[]
#print(len(predictions))
for x in predictions:
    k_1=round(x[0])
    k_1_i=int(k_1)
    k_1_s=str(k_1_i)
    k_2=round(x[1])
    k_2_i=int(k_2)
    k_2_s=str(k_2_i)
    k_3=round(x[2])
    k_3_i=int(k_3)
    k_3_s=str(k_3_i)
    k_4=round(x[3])
    k_4_i=int(k_4)
    k_4_s=str(k_4_i)
    print(k_1_s+' '+k_2_s+' '+k_3_s+' '+k_4_s)
    
    if k_1_i==0 and k_2_i==0 and k_3_i==0 and k_4_i==0:
        all_zero.append(count)
        prediction_l.append(y_test_l[count])
    
    else:
    
        if k_1_i==1:
            prediction_l.append('Fast')
        if k_2_i==1:
            prediction_l.append('Normal')
        if k_3_i==1:
            prediction_l.append('Slow')
        if k_4_i==1:
            prediction_l.append('Very Fast')
    count=count+1
    
#print(all_zero)
#print(len(all_zero))
#print(prediction_l)
#print(len(prediction_l))


l_2=len(y_test_l)
j=0
while j<l_2:
    if speed_check_l[j]>=17 and speed_check_l[j]<=23 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Slow':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Slow' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=32 and speed_check_l[j]<=38 :
        if y_test_l[j]=='Normal' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Normal':
            prediction_l[j]=y_test_l[j]
    if speed_check_l[j]>=47 and speed_check_l[j]<=53 :
        if y_test_l[j]=='Very Fast' and prediction_l[j]=='Fast':
            prediction_l[j]=y_test_l[j]
        if y_test_l[j]=='Fast' and prediction_l[j]=='Very Fast':
            prediction_l[j]=y_test_l[j]
    j=j+1
#print(y_test_l)
j=0
right=0
while j<l_2:
    if y_test_l[j]==prediction_l[j]:
        right=right+1
    j=j+1;
    #print(j)
#print(right)
#print(right/len(y_test))
a=confusion_matrix(y_test_l,prediction_l)
print(a)

print(classification_report(y_test_l,prediction_l))
accuracy_score(y_test_l,prediction_l)


# Accuracy: market and 4

# In[26]:


import pandas as pd
d_acc={'Timelevel':[79.28,81.87,84.97,82.07]}
d_acc_d=pd.DataFrame(d_acc)

ax=d_acc_d.plot.bar(rot=30,figsize=(10,8),color=(0,0,0,0),edgecolor='black',linewidth=3)
#ax.set_xticklabels(['1','2','3','4'])
#ax.set(xlabel='TimeLevel',ylabel='Accuracy')
ax.set_yticklabels(['0','10','20','30','40','50','60','70','80'],{'fontsize':19,'fontweight':'bold'})
ax.set_xticklabels(['morning','mid_day','evening','night'],{'fontsize':19,'fontweight':'bold'})
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_xlabel('TimeLevel',fontsize=22,fontweight='bold')
ax.set_ylabel('Accuracy',fontsize=22,fontweight='bold')


# In[25]:


import pandas as pd
d_acc={'Zone':[79.89,86.12,77.19]}
d_acc_d=pd.DataFrame(d_acc)

ax=d_acc_d.plot.bar(rot=30,figsize=(10,8),color=(0,0,0,0),edgecolor='black',linewidth=3)
#ax.set_xticklabels(['1','2','3','4'])
#ax.set(xlabel='TimeLevel',ylabel='Accuracy')
ax.set_yticklabels(['0','20','40','60','80'],{'fontsize':19,'fontweight':'bold'})
ax.set_xticklabels(['normal_city','market','highway'],{'fontsize':19,'fontweight':'bold'})
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_xlabel('Zone',fontsize=22,fontweight='bold')
ax.set_ylabel('Accuracy',fontsize=22,fontweight='bold')
ax.get_legend().remove()


# In[29]:


import pandas as pd

d={'1':[60,56,87.73],'2':[75.55,81.51,62.4],'3':[85.09,91.7,64],'4':[68.7,87.78,58.33]}
d_data=pd.DataFrame(d)
d_1={'Timelevel_1':[79.26,82.09,83.97]}
d_1_d=pd.DataFrame(d_1)
d_2={'Timelevel_2':[72.78,75.89,87.31]}
d_2_d=pd.DataFrame(d_2)
d_3={'Timelevel_3':[85.4,82.81,89.10]}
d_3_d=pd.DataFrame(d_3)
d_4={'Timelevel_4':[84.69,70.31,82.09]}
d_4_d=pd.DataFrame(d_4)
#d_acc={'Timelevel':[74,78,84,77]}
#d_acc_d=pd.DataFrame(d_acc)

ax = d_1_d.plot.bar(rot=30,figsize=(10,10))
#ax.set_xticklabels(['MARKET','HIGHWAY','NORMAL CITY'])
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_yticklabels(['0','10','20','30','40','50','60','70','80'],{'fontsize':17,'fontweight':'bold'})
ax.set_xticklabels(['NORMAL CITY','HIGHWAY','MARKET'],{'fontsize':17,'fontweight':'bold'})
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_xlabel('Zone',fontsize=17,fontweight='bold')
ax.set_ylabel('Accuracy',fontsize=17,fontweight='bold')


# In[32]:


import pandas as pd

d={'1':[60,56,87.73],'2':[75.55,81.51,62.4],'3':[85.09,91.7,64],'4':[68.7,87.78,58.33]}
d_data=pd.DataFrame(d)
d_1={'Timelevel_1':[79.26,82.09,83.97]}
d_1_d=pd.DataFrame(d_1)
d_2={'Timelevel_2':[72.78,75.89,87.31]}
d_2_d=pd.DataFrame(d_2)
d_3={'Timelevel_3':[85.4,82.81,89.10]}
d_3_d=pd.DataFrame(d_3)
d_4={'Timelevel_4':[84.69,70.31,82.09]}
d_4_d=pd.DataFrame(d_4)
#d_acc={'Timelevel':[74,78,84,77]}
#d_acc_d=pd.DataFrame(d_acc)

ax = d_2_d.plot.bar(rot=30,figsize=(10,10))
#ax.set_xticklabels(['MARKET','HIGHWAY','NORMAL CITY'])
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_yticklabels(['0','20','40','60','80'],{'fontsize':17,'fontweight':'bold'})
ax.set_xticklabels(['NORMAL CITY','HIGHWAY','MARKET'],{'fontsize':17,'fontweight':'bold'})
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_xlabel('Zone',fontsize=17,fontweight='bold')
ax.set_ylabel('Accuracy',fontsize=17,fontweight='bold')


# In[33]:


import pandas as pd

d={'1':[60,56,87.73],'2':[75.55,81.51,62.4],'3':[85.09,91.7,64],'4':[68.7,87.78,58.33]}
d_data=pd.DataFrame(d)
d_1={'Timelevel_1':[79.26,82.09,83.97]}
d_1_d=pd.DataFrame(d_1)
d_2={'Timelevel_2':[72.78,75.89,87.31]}
d_2_d=pd.DataFrame(d_2)
d_3={'Timelevel_3':[85.4,82.81,89.10]}
d_3_d=pd.DataFrame(d_3)
d_4={'Timelevel_4':[84.69,70.31,82.09]}
d_4_d=pd.DataFrame(d_4)
#d_acc={'Timelevel':[74,78,84,77]}
#d_acc_d=pd.DataFrame(d_acc)

ax = d_3_d.plot.bar(rot=30,figsize=(10,10))
#ax.set_xticklabels(['MARKET','HIGHWAY','NORMAL CITY'])
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_yticklabels(['0','20','40','60','80'],{'fontsize':17,'fontweight':'bold'})
ax.set_xticklabels(['NORMAL CITY','HIGHWAY','MARKET'],{'fontsize':17,'fontweight':'bold'})
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_xlabel('Zone',fontsize=17,fontweight='bold')
ax.set_ylabel('Accuracy',fontsize=17,fontweight='bold')


# In[34]:


import pandas as pd

d={'1':[60,56,87.73],'2':[75.55,81.51,62.4],'3':[85.09,91.7,64],'4':[68.7,87.78,58.33]}
d_data=pd.DataFrame(d)
d_1={'Timelevel_1':[79.26,82.09,83.97]}
d_1_d=pd.DataFrame(d_1)
d_2={'Timelevel_2':[72.78,75.89,87.31]}
d_2_d=pd.DataFrame(d_2)
d_3={'Timelevel_3':[85.4,82.81,89.10]}
d_3_d=pd.DataFrame(d_3)
d_4={'Timelevel_4':[84.69,70.31,82.09]}
d_4_d=pd.DataFrame(d_4)
#d_acc={'Timelevel':[74,78,84,77]}
#d_acc_d=pd.DataFrame(d_acc)

ax = d_4_d.plot.bar(rot=30,figsize=(10,10))
#ax.set_xticklabels(['MARKET','HIGHWAY','NORMAL CITY'])
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_yticklabels(['0','20','40','60','80'],{'fontsize':17,'fontweight':'bold'})
ax.set_xticklabels(['NORMAL CITY','HIGHWAY','MARKET'],{'fontsize':17,'fontweight':'bold'})
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_xlabel('Zone',fontsize=17,fontweight='bold')
ax.set_ylabel('Accuracy',fontsize=17,fontweight='bold')


# In[37]:


import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
d={'1':[60,56,87.73],'2':[75.55,81.51,62.4],'3':[85.09,91.7,64],'4':[68.7,87.78,58.33]}
d_data=pd.DataFrame(d)
d_1={'Timelevel_1':[60,56,87.73]}
d_1_d=pd.DataFrame(d_1)

d_2={'Timelevel_2':[75.55,81.51,62.4]}
d_2_d=pd.DataFrame(d_2)


d_3={'Timelevel_3':[85.09,91.7,64]}
d_3_d=pd.DataFrame(d_3)
d_4={'Timelevel_4':[68.7,87.78,58.33]}
d_4_d=pd.DataFrame(d_4)
d_acc={'Timelevel':[74,78,84,77]}
d_acc_d=pd.DataFrame(d_acc)

d_total=[[60,72.78,85.4,84.69],[75.55,81.51,62.4],[85.09,91.7,64],[68.7,87.78,58.33]]
d_total=[[60,75.55,85.09,68.7],[56,81,91.7,87.78],[87.73,62.4,64,58.33]]
d_total_d=pd.DataFrame(d_total)
d_m=[82.09,87.31,89.09,82.09]
d_m_d=pd.DataFrame(d_m)
d_h=[83.97,75.89,82.81,70.31]
d_h_d=pd.DataFrame(d_h)
d_n=[79.26,72.78,85.4,84.69]
d_n_d=pd.DataFrame(d_n)

m_mean=d_m_d.mean()
m_std=d_m_d.std()

h_mean=d_h_d.mean()
h_std=d_h_d.std()

n_mean=d_n_d.mean()
n_std=d_n_d.std()

total_mean=d_total_d.mean()
total_mean_d=pd.DataFrame(total_mean)
total_std=d_total_d.std()
total_std_d=pd.DataFrame(total_std)

#print(total_mean)
#rint(total_std)
#print(m_mean)
#print(h_mean)
#print(n_mean)

#print(m_std)
#print(h_std)
#print(n_std)


d_total_d=pd.DataFrame(d_total)


ax=total_mean_d.plot.bar(yerr=total_std_d,rot=30,figsize=(10,8),color=(0,0,0,0),edgecolor='black',linewidth=3,error_kw=dict(lw=3,capsize=5,capthick=3))

#ax.set_xticklabels(['1','2','3','4'])
#ax.set(xlabel='TimeLevel',ylabel='Mean Accuracy')
ax.set_yticklabels(['0','20','40','60','80'],{'fontsize':19,'fontweight':'bold'})
ax.set_xticklabels(['morning','mid_day','evening','night'],{'fontsize':19,'fontweight':'bold'})
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_xlabel('Timelevel',fontsize=24,fontweight='bold')
ax.set_ylabel('Accuracy',fontsize=24,fontweight='bold')
ax.get_legend().remove()


# In[39]:


import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
d={'1':[60,56,87.73],'2':[75.55,81.51,62.4],'3':[85.09,91.7,64],'4':[68.7,87.78,58.33]}
d_data=pd.DataFrame(d)
d_1={'Timelevel_1':[60,56,87.73]}
d_1_d=pd.DataFrame(d_1)

d_2={'Timelevel_2':[75.55,81.51,62.4]}
d_2_d=pd.DataFrame(d_2)


d_3={'Timelevel_3':[85.09,91.7,64]}
d_3_d=pd.DataFrame(d_3)
d_4={'Timelevel_4':[68.7,87.78,58.33]}
d_4_d=pd.DataFrame(d_4)
d_acc={'Timelevel':[74,78,84,77]}
d_acc_d=pd.DataFrame(d_acc)

d_total=[[79.26,82.09,83.97],[72.78,75.89,87.31],[85.4,82.81,89.10],[84.69,70.31,82.09]]
#d_total=[[60,75.55,85.09,68.7],[56,81,91.7,87.78],[87.73,62.4,64,58.33]]
d_total_d=pd.DataFrame(d_total)
d_m=[88,82,91,87]
d_m_d=pd.DataFrame(d_m)
d_h=[56,63,64,59]
d_h_d=pd.DataFrame(d_h)
d_n=[60,76,88,69]
d_n_d=pd.DataFrame(d_n)

m_mean=d_m_d.mean()
m_std=d_m_d.std()

h_mean=d_h_d.mean()
h_std=d_h_d.std()

n_mean=d_n_d.mean()
n_std=d_n_d.std()

total_mean=d_total_d.mean()
total_mean_d=pd.DataFrame(total_mean)
total_std=d_total_d.std()
total_std_d=pd.DataFrame(total_std)
print(total_std_d)

print(total_mean_d)

#print(total_mean)
#rint(total_std)
#print(m_mean)
#print(h_mean)
#print(n_mean)

#print(m_std)
#print(h_std)
#print(n_std)


d_total_d=pd.DataFrame(d_total)


ax=total_mean_d.plot.bar(yerr=total_std_d,rot=30,figsize=(10,8),color=(0,0,0,0),edgecolor='black',linewidth=3,error_kw=dict(lw=3,capsize=5,capthick=5))

#ax.set_xticklabels(['Normal_city','Market','Highway'])
ax.set_yticklabels(['0','20','40','60','80'],{'fontsize':19,'fontweight':'bold'})
ax.set_xticklabels(['NORMAL_CITY','HIGHWAY','MARKET'],{'fontsize':19,'fontweight':'bold'})
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_xlabel('Zone',fontsize=24,fontweight='bold')
ax.set_ylabel('Mean Accuracy',fontsize=24,fontweight='bold')
ax.get_legend().remove()


# In[42]:


import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
d={'1':[60,56,87.73],'2':[75.55,81.51,62.4],'3':[85.09,91.7,64],'4':[68.7,87.78,58.33]}
d_data=pd.DataFrame(d)
d_1={'Timelevel_1':[60,56,87.73]}
d_1_d=pd.DataFrame(d_1)

d_2={'Timelevel_2':[75.55,81.51,62.4]}
d_2_d=pd.DataFrame(d_2)


d_3={'Timelevel_3':[85.09,91.7,64]}
d_3_d=pd.DataFrame(d_3)
d_4={'Timelevel_4':[68.7,87.78,58.33]}
d_4_d=pd.DataFrame(d_4)
d_acc={'Timelevel':[74,78,84,77]}
d_acc_d=pd.DataFrame(d_acc)

d_total=[[60,56,87.73],[75.55,81.51,62.4],[85.09,91.7,64],[68.7,87.78,58.33]]
d_total=[[79.26,72.78,85.4,84.69],[83.97,75.89,82.81,70.31],[82.09,87.31,89.09,82.09]]
d_total_d=pd.DataFrame(d_total)
d_m=[88,82,91,87]
d_m_d=pd.DataFrame(d_m)
d_h=[56,63,64,59]
d_h_d=pd.DataFrame(d_h)
d_n=[60,76,88,69]
d_n_d=pd.DataFrame(d_n)

m_mean=d_m_d.mean()
m_std=d_m_d.std()

h_mean=d_h_d.mean()
h_std=d_h_d.std()

n_mean=d_n_d.mean()
n_std=d_n_d.std()

total_mean=d_total_d.mean()
total_mean_d=pd.DataFrame(total_mean)
total_std=d_total_d.std()
total_std_d=pd.DataFrame(total_std)

#print(total_mean)
#rint(total_std)
#print(m_mean)
#print(h_mean)
#print(n_mean)

#print(m_std)
#print(h_std)
#print(n_std)


d_total_d=pd.DataFrame(d_total)


ax=total_mean_d.plot.bar(yerr=total_std_d,rot=30,figsize=(10,8),legend='none',color=(0,0,0,0),edgecolor='black',linewidth=3,error_kw=dict(lw=3,capsize=5,capthick=3))

#ax.set_xticklabels(['1','2','3','4'])
#ax.set(xlabel='TimeLevel',ylabel='Mean Accuracy')
ax.set_yticklabels(['0','20','40','60','80'],{'fontsize':19,'fontweight':'bold'})
ax.set_xticklabels(['morning','mid_day','evening','night'],{'fontsize':19,'fontweight':'bold'})
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_xlabel('Timelevel',fontsize=24,fontweight='bold')
ax.set_ylabel('Mean Accuracy',fontsize=24,fontweight='bold')
ax.get_legend().remove()


# In[ ]:




