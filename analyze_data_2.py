#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

df=pd.read_csv('6mar.csv')

accu=df[df.Timelevel==1]
accu_1_d=pd.DataFrame(accu)

print(accu_1_d)


# In[6]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[df.Timelevel==1]
accu_1_d=pd.DataFrame(accu)

print(len(accu_1_d))
X=accu_1_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_1_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[ ]:


# from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[df.Timelevel==2]
accu_2_d=pd.DataFrame(accu)

print(len(accu_2_d))

X=accu_2_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_2_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[8]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[df.Timelevel==3]
accu_3_d=pd.DataFrame(accu)

print(len(accu_3_d))

X=accu_3_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_3_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[9]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[df.Timelevel==4]
accu_4_d=pd.DataFrame(accu)

print(len(accu_4_d))

X=accu_4_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_4_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[10]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[df.Zone=='highway']
accu_highway_d=pd.DataFrame(accu)

print(len(accu_highway_d))

X=accu_highway_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_highway_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[11]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[df.Zone=='normal_city']
accu_nc_d=pd.DataFrame(accu)

print(len(accu_nc_d))

X=accu_nc_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_nc_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[12]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[df.Zone=='market']
accu_market_d=pd.DataFrame(accu)

print(len(accu_market_d))

X=accu_market_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_market_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[13]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[(df.Zone=='normal_city') & (df.Timelevel==1)]
accu_nc_1_d=pd.DataFrame(accu)

print(len(accu_nc_1_d))

X=accu_nc_1_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_nc_1_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[14]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[(df.Zone=='normal_city') & (df.Timelevel==2)]
accu_nc_2_d=pd.DataFrame(accu)

print(len(accu_nc_2_d))

X=accu_nc_2_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_nc_2_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[15]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[(df.Zone=='normal_city') & (df.Timelevel==3)]
accu_nc_3_d=pd.DataFrame(accu)

print(len(accu_nc_3_d))

X=accu_nc_3_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_nc_3_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[16]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[(df.Zone=='normal_city') & (df.Timelevel==4)]
accu_nc_4_d=pd.DataFrame(accu)

print(len(accu_nc_4_d))

X=accu_nc_4_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_nc_4_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[17]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[(df.Zone=='market') & (df.Timelevel==4)]
accu_nc_4_d=pd.DataFrame(accu)

print(len(accu_nc_4_d))

X=accu_nc_4_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_nc_4_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[18]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[(df.Zone=='market') & (df.Timelevel==3)]
accu_nc_4_d=pd.DataFrame(accu)

print(len(accu_nc_4_d))

X=accu_nc_4_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_nc_4_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[19]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[(df.Zone=='market') & (df.Timelevel==2)]
accu_nc_4_d=pd.DataFrame(accu)

print(len(accu_nc_4_d))

X=accu_nc_4_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_nc_4_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[20]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[(df.Zone=='market') & (df.Timelevel==2)]
accu_nc_4_d=pd.DataFrame(accu)

print(len(accu_nc_4_d))

X=accu_nc_4_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_nc_4_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[21]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[(df.Zone=='highway') & (df.Timelevel==4)]
accu_nc_4_d=pd.DataFrame(accu)

print(len(accu_nc_4_d))

X=accu_nc_4_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_nc_4_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[22]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[(df.Zone=='highway') & (df.Timelevel==2)]
accu_nc_4_d=pd.DataFrame(accu)

print(len(accu_nc_4_d))

X=accu_nc_4_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_nc_4_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[23]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[(df.Zone=='highway') & (df.Timelevel==3)]
accu_nc_4_d=pd.DataFrame(accu)

print(len(accu_nc_4_d))

X=accu_nc_4_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_nc_4_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[24]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[(df.Zone=='highway') & (df.Timelevel==1)]
accu_nc_4_d=pd.DataFrame(accu)

print(len(accu_nc_4_d))

X=accu_nc_4_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_nc_4_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[25]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df=pd.read_csv('6mar.csv')

accu=df[(df.Zone=='market') & (df.Timelevel==4)]
accu_nc_4_d=pd.DataFrame(accu)

print(len(accu_nc_4_d))

X=accu_nc_4_d[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=accu_nc_4_d[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
#print(X_test)
#print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()

value_check=new_y.tolist()
#print(value_check)

new_y_d=pd.DataFrame(new_y)
gbc=XGBClassifier(eta=0.5,max_depth=6,seed=25,sub_sample=0.7,nthread=4)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

#print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
y_pred_l=y_pred.tolist()      #
#print(y_pred_l)
#print(y_pred.type())
speed_check=y_test[1].copy()
#speed_check_d=pd.DataFrame(speed_check)
speed_check_l=speed_check.tolist()     #

l=len(speed_check_l)
#print(l)
i=0
while i<l:
    if speed_check_l[i]>=17 and speed_check_l[i]<=23 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32 and speed_check_l[i]<=38 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47 and speed_check_l[i]<=53 :
        if value_check[i]=='Very Fast' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Very Fast':
            y_pred_l[i]=value_check[i]
    i=i+1
y_pred_f=np.array(y_pred_l)        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred_f)
print(a)

print(classification_report(new_y_d,y_pred_f))
accuracy_score(new_y_d,y_pred_f)


# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns

time_accu=[('1',73.5),('2',77.66),('3',83.4),('4',76.97)]

time_acc_1=[60,56,87.73]

time_acc_2=[75.55,81.51,62.4]

time_acc_3=[85.09,91.7,64]

time_acc_4=[68.7,87.78,58.33]


# In[146]:



print("hi")


# In[59]:


import pandas as pd

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

ax = d_1_d.plot.bar(rot=0)
ax.set_xticklabels(['MARKET','HIGHWAY','NORMAL CITY'])
ax.set(xlabel='Zone',ylabel='Accuracy')
ax=d_2_d.plot.bar(rot=0)
ax.set_xticklabels(['MARKET','HIGHWAY','NORMAL CITY'])
ax.set(xlabel='Zone',ylabel='Accuracy')
ax=d_3_d.plot.bar(rot=0)
ax.set_xticklabels(['MARKET','HIGHWAY','NORMAL CITY'])
ax.set(xlabel='Zone',ylabel='Accuracy')
ax=d_4_d.plot.bar(rot=0)
ax.set_xticklabels(['MARKET','HIGHWAY','NORMAL CITY'])
ax.set(xlabel='Zone',ylabel='Accuracy')



ax=d_acc_d.plot.bar(rot=0)
ax.set_xticklabels(['1','2','3','4'])
ax.set(xlabel='TimeLevel',ylabel='Accuracy')
ax = d_data.plot.bar(rot=0)
ax.set_xticklabels(['MARKET','HIGHWAY','NORMAL CITY'])
ax.set(xlabel='Zone',ylabel='Accuracy')


# In[136]:


import pandas as pd

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

ax = d_1_d.plot.bar(rot=0,figsize=(10,10))
#ax.set_xticklabels(['MARKET','HIGHWAY','NORMAL CITY'])
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_yticklabels(['0','20','40','60','80'],{'fontsize':17,'fontweight':'bold'})
ax.set_xticklabels(['MARKET','HIGHWAY','NORMAL CITY'],{'fontsize':17,'fontweight':'bold'})
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_xlabel('Zone',fontsize=17,fontweight='bold')
ax.set_ylabel('Accuracy',fontsize=17,fontweight='bold')


# In[138]:


import pandas as pd

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

ax=d_2_d.plot.bar(rot=0,figsize=(10,10))
#ax.set_xticklabels(['MARKET','HIGHWAY','NORMAL CITY'])
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_yticklabels(['0','10','20','30','40','50','60','70','80'],{'fontsize':17,'fontweight':'bold'})
ax.set_xticklabels(['NORMAL_CITY','MARKET','HIGHWAY'],{'fontsize':17,'fontweight':'bold'})
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_xlabel('Zone',fontsize=17,fontweight='bold')
ax.set_ylabel('Accuracy',fontsize=17,fontweight='bold')


# In[135]:


import pandas as pd

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

ax=d_3_d.plot.bar(rot=0,figsize=(12,10))
#ax.set_xticklabels(['MARKET','HIGHWAY','NORMAL CITY'])
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_yticklabels(['0','20','40','60','80'],{'fontsize':17,'fontweight':'bold'})
ax.set_xticklabels(['NORMAL_CITY','MARKET','HIGHWAY'],{'fontsize':17,'fontweight':'bold'})
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_xlabel('Zone',fontsize=17,fontweight='bold')
ax.set_ylabel('Accuracy',fontsize=17,fontweight='bold')


# In[174]:


import pandas as pd

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

ax=d_acc_d.plot.bar(rot=0,figsize=(10,8),color=(0,0,0,0),edgecolor='black',linewidth=3)
#ax.set_xticklabels(['1','2','3','4'])
#ax.set(xlabel='TimeLevel',ylabel='Accuracy')
ax.set_yticklabels(['0','10','20','30','40','50','60','70','80'],{'fontsize':19,'fontweight':'bold'})
ax.set_xticklabels(['1','2','3','4'],{'fontsize':19,'fontweight':'bold'})
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_xlabel('TimeLevel',fontsize=22,fontweight='bold')
ax.set_ylabel('Accuracy',fontsize=22,fontweight='bold')


# In[189]:


import pandas as pd

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

ax = d_data.plot.bar(rot=30,figsize=(12,10))
ax.set_yticklabels(['0','10','20','30','40','50','60','70','80'],{'fontsize':17,'fontweight':'bold'})
ax.set_xticklabels(['MARKET','HIGHWAY','NORMAL CITY'],{'fontsize':17,'fontweight':'bold'})
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_xlabel('Zone',fontsize=24,fontweight='bold')
ax.set_ylabel('Accuracy',fontsize=24,fontweight='bold')


# In[186]:


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

print(m_mean)
print(h_mean)
print(n_mean)

print(m_std)
print(h_std)
print(n_std)


d_total_d=pd.DataFrame(d_total)


ax=d_acc_d.plot.bar(rot=0,figsize=(10,8),fontsize=17,color=(0,0,0,0),edgecolor='black',linewidth=3)
ax.set_yticklabels(['0','10','20','30','40','50','60','70','80'],{'fontsize':19,'fontweight':'bold'})
ax.set_xticklabels(['1','2','3','4'],{'fontsize':19,'fontweight':'bold'})
ax.set_xlabel('TimeLevel',fontsize=24,fontweight='bold')
ax.set_ylabel('Accuracy',fontsize=24,fontweight='bold')
ax.get_legend().remove()


# In[68]:


from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
data_df=pd.read_csv('6mar.csv')

x=data_df['Honk_duration'].values
y=data_df['Mean_speed_kmph'].values
labels = data_df['Zone'].values
df = pd.DataFrame(dict(x=x, y=y, label=labels))
groups = df.groupby('label')

# Plot
fig, ax = plt.subplots(figsize=(15,8))
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
s= 500
for name, group in groups:
    ax.scatter(group.x, group.y, marker='.',s=s, label=name)
    s = s- 150

leg = plt.legend(('HIGHWAY','MARKET','NORMAL CITY'),fontsize ='xx-large')
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_linewidth(2)

for i in [20,35,50]:
  plt.axhline(y=i,linestyle=':',linewidth=5,color='k')
ax.set_xlabel('Honk duration (sec)',fontsize = 22,fontweight='bold')
ax.set_ylabel('Mean speed (kmph)',fontsize = 22,fontweight='bold')
plt.xticks(np.arange(0, max(x)+3, 4),fontsize = 20,fontweight='bold')
plt.yticks(np.arange(0, 90,10),fontsize = 20,fontweight='bold')

plt.grid(color='k', linestyle='-', linewidth=1,alpha=0.5)

sl = data_df['Honk_duration'][data_df['Class'] == 'Slow'].values
no = data_df['Honk_duration'][data_df['Class'] == 'Normal'].values
fa = data_df['Honk_duration'][data_df['Class'] == 'Fast'].values
vf = data_df['Honk_duration'][data_df['Class'] == 'Very Fast'].values
sl_mean = np.mean(sl)
no_mean = np.mean(no)
fa_mean = np.mean(fa)
vf_mean = np.mean(vf)
sl_std = np.std(sl)
no_std = np.std(no)
fa_std = np.std(fa)
vf_std = np.std(vf)
# Create lists for the plot
Classes = ['S', 'N', 'F','V']
x_pos = np.arange(len(Classes))
CHs = [sl_mean, no_mean, fa_mean, vf_mean]
error = [sl_std, no_std, fa_std, vf_std]
# Create inset of width 30% and height 30% of the parent axes' bounding box at the lower left corner (loc=3)
axins = inset_axes(ax, width="30%", height="30%", loc='upper center')
axins.bar(x_pos, CHs, align='center',yerr=error, alpha=1, ecolor='black')
#axins.set_yticklabels({'fontsize':17,'fontweight': 'bold'})
axins.set_xticklabels(['','S', 'N', 'F','V'],{'fontsize':17,'fontweight': 'bold'})
plt.show()


# In[180]:


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
d_total=[[60,75.55,85.09,68.7],[56,81,91.7,87.78],[87.73,62.4,64,58.33]]
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


ax=total_mean_d.plot.bar(yerr=total_std_d,rot=0,figsize=(10,8),color=(0,0,0,0),edgecolor='black',linewidth=3,error_kw=dict(lw=3,capsize=5,capthick=3))

#ax.set_xticklabels(['1','2','3','4'])
#ax.set(xlabel='TimeLevel',ylabel='Mean Accuracy')
ax.set_yticklabels(['0','20','40','60','80'],{'fontsize':19,'fontweight':'bold'})
ax.set_xticklabels(['1','2','3','4'],{'fontsize':19,'fontweight':'bold'})
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_xlabel('Timelevel',fontsize=24,fontweight='bold')
ax.set_ylabel('Accuracy',fontsize=24,fontweight='bold')
ax.get_legend().remove()


# In[2]:


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

d_total=[[60,88,56],[75.55,81.51,62.4],[85.09,91.7,64],[68.7,87.78,58.33]]
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
ax.set_xticklabels(['NORMAL_CITY','MARKET','HIGHWAY'],{'fontsize':19,'fontweight':'bold'})
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_xlabel('Zone',fontsize=24,fontweight='bold')
ax.set_ylabel('Mean Accuracy',fontsize=24,fontweight='bold')
ax.get_legend().remove()


# In[206]:


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
d_total=[[60,75.55,85.09,68.7],[56,81,91.7,87.78],[87.73,62.4,64,58.33]]
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


ax=total_mean_d.plot.bar(yerr=total_std_d,rot=0,figsize=(10,8),legend='none',color=(0,0,0,0),edgecolor='black',linewidth=3,error_kw=dict(lw=3,capsize=5,capthick=3))

#ax.set_xticklabels(['1','2','3','4'])
#ax.set(xlabel='TimeLevel',ylabel='Mean Accuracy')
ax.set_yticklabels(['0','20','40','60','80'],{'fontsize':19,'fontweight':'bold'})
ax.set_xticklabels(['1','2','3','4'],{'fontsize':19,'fontweight':'bold'})
#ax.set(xlabel='Zone',ylabel='Accuracy')
ax.set_xlabel('Timelevel',fontsize=24,fontweight='bold')
ax.set_ylabel('Mean Accuracy',fontsize=24,fontweight='bold')
ax.get_legend().remove()


# In[3]:


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

print(m_mean)
print(h_mean)
print(n_mean)

print(m_std)
print(h_std)
print(n_std)


d_total_d=pd.DataFrame(d_total)


ax=d_acc_d.plot.bar(rot=30,figsize=(10,8),fontsize=17,color=(0,0,0,0),edgecolor='black',linewidth=3)
ax.set_yticklabels(['0','10','20','30','40','50','60','70','80'],{'fontsize':19,'fontweight':'bold'})
ax.set_xticklabels(['Morning','Mid-day','Evening','Night'],{'fontsize':19,'fontweight':'bold'})
ax.set_xlabel('TimeLevel',fontsize=24,fontweight='bold')
ax.set_ylabel('Accuracy',fontsize=24,fontweight='bold')
ax.get_legend().remove()


# In[ ]:




