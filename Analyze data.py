#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[ ]:





# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=11,weights='distance')
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=14,weights='distance')
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=13,weights='distance')
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=8,weights='distance')
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['RMS','Intersection density','WiFi density']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=8,weights='distance')
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['RMS','Intersection density','WiFi density']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=11,weights='distance')
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['RMS','Intersection density','WiFi density']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['RMS','Intersection density','WiFi density']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['RMS','Intersection density','WiFi density']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=14)
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
road=pd.read_csv('dataset.csv')

road.head()
'''road_lookup=road.Zone.unique()'''
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
'''knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)
road_prediction=knn.predict([[181.365898,5,0,6.35698]])
road_lookup[road_prediction[0]]'''
k_range=np.arange(1,101)
train_accuracy=np.empty(len(k_range))
test_accuracy=np.empty(len(k_range))
for i,k in enumerate(k_range):
	knn=KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train,y_train)
	train_accuracy[i]=knn.score(X_train,y_train)
	test_accuracy[i]=knn.score(X_test,y_test)
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(k_range, test_accurac-y, label = 'Testing Accuracy')
plt.plot(k_range, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0,105,5))
plt.show()


# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
road=pd.read_csv('dataset.csv')

road.head()
'''road_lookup=road.Zone.unique()'''
X=road[['Intersection density','WiFi density','Honk_duration','RMS']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
'''knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)
road_prediction=knn.predict([[181.365898,5,0,6.35698]])
road_lookup[road_prediction[0]]'''
k_range=np.arange(1,101)
train_accuracy=np.empty(len(k_range))
test_accuracy=np.empty(len(k_range))
for i,k in enumerate(k_range):
	knn=KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train,y_train)
	train_accuracy[i]=knn.score(X_train,y_train)
	test_accuracy[i]=knn.score(X_test,y_test)
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(k_range, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_range, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0,105,5))
plt.show()


# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
road=pd.read_csv('dataset.csv')

road.head()
'''road_lookup=road.Zone.unique()'''
X=road[['Intersection density','WiFi density','Honk_duration','RMS']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
'''knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)
road_prediction=knn.predict([[181.365898,5,0,6.35698]])
road_lookup[road_prediction[0]]'''
k_range=np.arange(1,101)
train_accuracy=np.empty(len(k_range))
test_accuracy=np.empty(len(k_range))
for i,k in enumerate(k_range):
	knn=KNeighborsClassifier(n_neighbors=k,weights='distance')
	knn.fit(X_train,y_train)
	train_accuracy[i]=knn.score(X_train,y_train)
	test_accuracy[i]=knn.score(X_test,y_test)
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(k_range, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_range, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0,105,5))
plt.show()


# In[27]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)
# Instantiate a k-NN classifier: knn

list=pd.read_csv('dataset.csv')
x=['Honk_duration','RMS','Intersection density','WiFi density']
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[28]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)
# Instantiate a k-NN classifier: knn

list=pd.read_csv('dataset.csv')
x=['Honk_duration','RMS','Intersection density','WiFi density']
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
knn = KNeighborsClassifier(n_neighbors=11)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[29]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)
# Instantiate a k-NN classifier: knn

list=pd.read_csv('dataset.csv')
x=['Honk_duration','RMS','Intersection density','WiFi density']
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
knn = KNeighborsClassifier(n_neighbors=10)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[30]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)
# Instantiate a k-NN classifier: knn

list=pd.read_csv('dataset.csv')
x=['Honk_duration','RMS','Intersection density','WiFi density']
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
knn = KNeighborsClassifier(n_neighbors=10,weights='distance')

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['RMS','Intersection density','WiFi density']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=6,algorithm='kd_tree')
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['RMS','Intersection density','WiFi density']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=11,algorithm='kd_tree')
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['RMS','Intersection density','WiFi density']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=11,algorithm='ball_tree')
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['RMS','Intersection density','WiFi density','Honk_duration']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=11,algorithm='ball_tree')
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['RMS','Intersection density','WiFi density','Honk_duration']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=11,algorithm='kd_tree')
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[38]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['RMS','Intersection density','WiFi density','Honk_duration']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=11,algorithm='kd_tree')
knn.fit(X_train,y_train)
knn.score(X_train,y_train)
print("a")
#knn.score(X_test,y_test)


# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['RMS','Intersection density','WiFi count','Honk_duration']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=11,algorithm='kd_tree')
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['RMS','Intersection count','WiFi count','Honk_duration']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=11,algorithm='kd_tree')
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['Intersection count','WiFi count','Honk_duration']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=11,algorithm='kd_tree')
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['RMS','Intersection count','WiFi count','Honk_duration']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=11,algorithm='ball_tree')
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['RMS','Intersection count','WiFi count','Honk_duration']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[44]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['RMS','Intersection count','WiFi count','Honk_duration']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')
X=list[['RMS','Intersection count','WiFi count','Honk_duration']].values
Y=list['Class'].values

X_d=pd.DataFrame(X)
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d_2,test_size=0.4,random_state=42)

knn=KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train,y_train)
#knn.score(X_train,y_train)
print("a")
knn.score(X_test,y_test)


# In[46]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)
# Instantiate a k-NN classifier: knn

list=pd.read_csv('dataset.csv')
x=['Honk_duration','RMS','Intersection density','WiFi density']
X=list[['Honk_duration','RMS','Intersection count','WiFi count']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
knn = KNeighborsClassifier(n_neighbors=11)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[47]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
road=pd.read_csv('dataset.csv')

road.head()
'''road_lookup=road.Zone.unique()'''
X=road[['Intersection count
 [0. 1. 0. 0.]
 [0. 1. 0. 0.]
 ...
 [0. 1. 0. 0.]
 [0. 0. 0. 1.]
 [0. 1. 0. 0.]]
0.6613508442776735
￼','WiFi count','Honk_duration','RMS']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
'''knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)
road_prediction=knn.predict([[181.365898,5,0,6.35698]])
road_lookup[road_prediction[0]]'''
k_range=np.arange(1,101)
train_accuracy=np.empty(len(k_range))
test_accuracy=np.empty(len(k_range))
for i,k in enumerate(k_range):
	knn=KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train,y_train)
	train_accuracy[i]=knn.score(X_train,y_train)
	test_accuracy[i]=knn.score(X_test,y_test)
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(k_range, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_range, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0,105,5))
plt.show()


# In[4]:


import pandas as pd
import numpy as np
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(Y)
X_train,X_test,y_train,y_test=train_test_split(X_d,Y_d,test_size=0.25,random_state=21)
clf = DecisionTreeClassifier(max_depth=9,min_samples_leaf=6)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))


# In[6]:


import pandas as pd
import numpy as np
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi count']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(Y)
X_train,X_test,y_train,y_test=train_test_split(X_d,Y_d,test_size=0.25,random_state=21)
clf = DecisionTreeClassifier(max_depth=9,min_samples_leaf=6)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))


# In[52]:


import pandas as pd
import numpy as np
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi count']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(y)
X_train,X_test,y_train,y_test=train_test_split(X_d,Y_d,test_size=0.4,random_state=21)
clf = DecisionTreeClassifier(max_depth=9,min_samples_leaf=6)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))


# In[54]:


import pandas as pd
import numpy as np
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi count']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(y)
X_train,X_test,y_train,y_test=train_test_split(X_d,Y_d,test_size=0.2,random_state=21)
clf = DecisionTreeClassifier(max_depth=10,min_samples_leaf=6)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))


# In[55]:


import pandas as pd
import numpy as np
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi count']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(y)
X_train,X_test,y_train,y_test=train_test_split(X_d,Y_d,test_size=0.2,random_state=21)
clf = DecisionTreeClassifier(max_depth=9,min_samples_leaf=10)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))


# In[56]:


import pandas as pd
import numpy as np
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi count']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(y)
X_train,X_test,y_train,y_test=train_test_split(X_d,Y_d,test_size=0.4,random_state=21)
clf = DecisionTreeClassifier(max_depth=9,min_samples_leaf=5)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))


# In[57]:


import pandas as pd
import numpy as np
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi count']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(y)
X_train,X_test,y_train,y_test=train_test_split(X_d,Y_d,test_size=0.4,random_state=21)
clf = DecisionTreeClassifier(max_depth=9,min_samples_leaf=12)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))


# In[58]:





# In[61]:



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier


list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection count','WiFi count']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)
#print("c")
X_train,X_test,y_train,y_test=train_test_split(X_d_2,y_d_2,test_size=0.1,random_state=21)
#print("d")
clf = DecisionTreeClassifier(max_depth=8,min_samples_leaf=8)
#print("e")
knn.fit(X_train,y_train)
#print("f")

y_pred=knn.predict(X_test)
#print("g")
#print(y_pred)
print(knn.score(X_test,y_test))


# In[62]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)
# Instantiate a k-NN classifier: knn

list=pd.read_csv('dataset.csv')
x=['Honk_duration','RMS','Intersection density','WiFi density']
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
clf = DecisionTreeClassifier(max_depth=8,min_samples_leaf=8)

# Fit the classifier to the training data
clf.fit(X_train,y_train)

# Predict the labels of the test data: y_pred
y_pred = clf.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[63]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)
# Instantiate a k-NN classifier: knn

list=pd.read_csv('dataset.csv')
x=['Honk_duration','RMS','Intersection density','WiFi density']
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
clf = DecisionTreeClassifier(max_depth=9,min_samples_leaf=6)

# Fit the classifier to the training data
clf.fit(X_train,y_train)

# Predict the labels of the test data: y_pred
y_pred = clf.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[64]:


import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)
#print("a")
X_d_2=pd.get_dummies(X_d)
#print("b")
y_d_2=pd.get_dummies(y_d)
#print("c")
#X_train,X_test,y_train,y_test=train_test_split(X_d_2,y_d_2,test_size=0.1,random_state=21)
#print("d")
knn = KNeighborsClassifier(n_neighbors=11)
#print("e")
cv_scores = cross_val_score(knn, X_d_2, y_d_2,cv=8)
print(np.mean(cv_scores))


# In[65]:


import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier


list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)
#print("a")
X_d_2=pd.get_dummies(X_d)
#print("b")
y_d_2=pd.get_dummies(y_d)
#print("c")
#X_train,X_test,y_train,y_test=train_test_split(X_d_2,y_d_2,test_size=0.1,random_state=21)
#print("d")
knn = KNeighborsClassifier(n_neighbors=11)
#print("e")
cv_scores = cross_val_score(knn, X_d_2, y_d_2,cv=4)
print(np.mean(cv_scores))


# In[72]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=20)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[74]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=200)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[75]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[76]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=150)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[77]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=170)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[78]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=180)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[79]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=175)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[80]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi count']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=180)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[81]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi count']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=150)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[82]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi count']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=220)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[83]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=220)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[84]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=180)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[115]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=150)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[116]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=130)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[87]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=170,max_depth=6)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[88]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=170,max_depth=10)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[89]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=170,max_depth=15)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[90]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=170,max_depth=12)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[99]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=170,max_depth=10,min_samples_split=5)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[92]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=170,max_depth=10,min_samples_split=9)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[93]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=170,max_depth=10,min_samples_split=8)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[94]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=170,max_depth=10,min_samples_split=7)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[95]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=170,max_depth=10,min_samples_split=4)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[96]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=170,max_depth=10,min_samples_split=5,min_samples_leaf=4)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[97]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=170,max_depth=10,min_samples_split=5,min_samples_leaf=2)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[98]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=170,max_depth=10,min_samples_split=5)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[103]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.2,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=170,max_depth=10,min_samples_split=5)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))
feature_imp = pd.Series(clf.feature_importances_,index=['Honk_duration','RMS','Intersection count','WiFi density']).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[106]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)
# Instantiate a k-NN classifier: knn

list=pd.read_csv('dataset.csv')
x=['Honk_duration','RMS','Intersection density','WiFi density']
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
clf=RandomForestClassifier(n_estimators=170,max_depth=10,min_samples_split=5)

# Fit the classifier to the training data
clf.fit(X_train,y_train)

# Predict the labels of the test data: y_pred
y_pred = clf.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[117]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list.drop(['Class','Time'],axis=1).values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d_2,Y_d_2,test_size=0.4,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.3,n_estimators=100,max_depth=6,min_samples_split=3,min_samples_leaf=3)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))



# In[123]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d,test_size=0.4,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.3,n_estimators=100,max_depth=6,min_samples_split=3,min_samples_leaf=3)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))



# In[124]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d,test_size=0.4,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.3,n_estimators=100,max_depth=6,min_samples_split=3,min_samples_leaf=3)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))



# In[125]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection count','WiFi count']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d,test_size=0.4,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.3,n_estimators=100,max_depth=6,min_samples_split=3,min_samples_leaf=3)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))



# In[127]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d,test_size=0.4,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.3,n_estimators=100,max_depth=6,min_samples_split=3,min_samples_leaf=3)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))



# In[128]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d,test_size=0.4,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.2,n_estimators=100,max_depth=6,min_samples_split=3,min_samples_leaf=3)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))



# In[129]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d,test_size=0.4,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.1,n_estimators=100,max_depth=6,min_samples_split=3,min_samples_leaf=3)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))



# In[130]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d,test_size=0.4,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.1,n_estimators=120,max_depth=6,min_samples_split=3,min_samples_leaf=3)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))



# In[131]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d,test_size=0.4,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.1,n_estimators=130,max_depth=6,min_samples_split=3,min_samples_leaf=3)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))



# In[132]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d,test_size=0.4,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.1,n_estimators=125,max_depth=6,min_samples_split=3,min_samples_leaf=3)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))



# In[135]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d,test_size=0.4,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.1,n_estimators=120,max_depth=8,min_samples_split=3,min_samples_leaf=3)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))



# In[136]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d,test_size=0.4,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.1,n_estimators=120,max_depth=7,min_samples_split=3,min_samples_leaf=3)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))



# In[137]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d,test_size=0.4,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.1,n_estimators=120,max_depth=6,min_samples_split=5,min_samples_leaf=3)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))



# In[138]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d,test_size=0.4,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.1,n_estimators=120,max_depth=6,min_samples_split=2,min_samples_leaf=3)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))



# In[4]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,Y_d,test_size=0.4,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.1,n_estimators=120,max_depth=6,min_samples_split=2,min_samples_leaf=3)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))
#y_pred = clf.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[8]:


from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

list=pd.read_csv('6mar.csv')

cond=list['Mean_speed_kmph'].values
cond_d=pd.DataFrame(cond)
X=list[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list[['SpeedRange']].values
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
#print(y_d_2)
#print(cond_d)
y_d.loc[cond_d[0]<=22.5,'SpeedRange']=['1']
y_d.loc[cond_d[0]>22.5,'SpeedRange']=['2']
y_d.loc[cond_d[0]>37.5,'SpeedRange']=['3']
y_d.loc[cond_d[0]>52.5,'SpeedRange']=['4']
i=0
#range=5328
while i<5328:
    if cond_d.iloc[i,0]>=18.5 and cond_d.iloc[i,0]<22.5:
        y_d.at[i,'SpeedRange']=['1','2']
    if cond_d.iloc[i,0]>=32.5 and cond_d.iloc[i,0]<37.5:
        y_d.at[i,'SpeedRange']=['2','3']
    if cond_d.iloc[i,0]>=47.5 and cond_d.iloc[i,0]<52.5:
        y_d.at[i,'SpeedRange']=['3','4']
    i=i+1
    
    
#print(y_d['SpeedRange'])
new_y=y_d['SpeedRange'].copy()
new_y_d=pd.DataFrame(new_y)
#print(new_y)
y_l=new_y.values.tolist()

#print(y_l)
#print(y_l.type())
new_y_d2=MultiLabelBinarizer().fit_transform(y_l)
#print(new_y_d2)

X_train,X_test,y_train,y_test=train_test_split(X_d,new_y_d2,test_size=0.4,random_state=21)
clf = DecisionTreeClassifier(max_depth=9,min_samples_leaf=6)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))





# In[93]:


from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
y = [['2','3', '4'], ['2'], ['0', '1', '3'],['2'], ['0', '1','2', '3', '4'], ['0', '1', '2']]
df=pd.DataFrame(y)
print(y.type())
MultiLabelBinarizer().fit_transform(y)


# In[21]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
print(X_test)
print(y_test)
gbc=GradientBoostingClassifier(learning_rate=0.1,n_estimators=120,max_depth=6,min_samples_split=2,min_samples_leaf=3)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))
y_pred = gbc.predict(X_test)
print(y_pred)
# Generate the confusion matrix and classification report
a=confusion_matrix(y_test,y_pred)
print(a)


# In[29]:


#from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

list=pd.read_csv('6mar.csv')

cond=list['Mean_speed_kmph'].values
cond_d=pd.DataFrame(cond)
X=list[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list[['SpeedRange']].values
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
#print(y_d_2)
#print(cond_d)
y_d.loc[cond_d[0]<=22.5,'SpeedRange']=['1']
y_d.loc[cond_d[0]>22.5,'SpeedRange']=['2']
y_d.loc[cond_d[0]>37.5,'SpeedRange']=['3']
y_d.loc[cond_d[0]>52.5,'SpeedRange']=['4']
i=0
#range=5328
while i<5328:
    if cond_d.iloc[i,0]>=18.5 and cond_d.iloc[i,0]<22.5:
        y_d.at[i,'SpeedRange']=['1','2']
    if cond_d.iloc[i,0]>=32.5 and cond_d.iloc[i,0]<37.5:
        y_d.at[i,'SpeedRange']=['2','3']
    if cond_d.iloc[i,0]>=47.5 and cond_d.iloc[i,0]<52.5:
        y_d.at[i,'SpeedRange']=['3','4']
    i=i+1
    
    
#print(y_d['SpeedRange'])
new_y=y_d['SpeedRange'].copy()
new_y_d=pd.DataFrame(new_y)
#print(new_y)
y_l=new_y.values.tolist()

#print(y_l)
#print(y_l.type())
new_y_d2=MultiLabelBinarizer().fit_transform(y_l)
#print(new_y_d2)

X_train,X_test,y_train,y_test=train_test_split(X_d,new_y_d2,test_size=0.4,random_state=21)
clf = DecisionTreeClassifier(max_depth=9,min_samples_leaf=6)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

y_pred = clf.predict(X_test)
print(y_pred)
# Generate the confusion matrix and classification report
a=multilabel_confusion_matrix(y_test, y_pred)
print(a)



# In[23]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('6mar.csv')

X=list[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
print(X_test)
print(y_test)
gbc=GradientBoostingClassifier(learning_rate=0.1,n_estimators=120,max_depth=6,min_samples_split=2,min_samples_leaf=3)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))
y_pred = gbc.predict(X_test)
print(y_pred)
# Generate the confusion matrix and classification report
a=confusion_matrix(y_test,y_pred)
print(a)


# In[27]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('6mar.csv')

X=list[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)
print(X_test)
print(y_test)
new_y_2=y_train[0].copy()
new_y_d_2=pd.DataFrame(new_y_2)
new_y=y_test[0].copy()
new_y_d=pd.DataFrame(new_y)
gbc=GradientBoostingClassifier(learning_rate=0.1,n_estimators=120,max_depth=6,min_samples_split=2,min_samples_leaf=3)

gbc.fit(X_train,new_y_d_2)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,new_y_d))
y_pred = gbc.predict(X_test)
print(y_pred)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred)
print(a)
#new_y=y_d['SpeedRange'].copy()
#new_y_d=pd.DataFrame(new_y)


# In[37]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('6mar.csv')

X=list[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list[['Class','Mean_speed_kmph']].values
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
gbc=GradientBoostingClassifier(learning_rate=0.1,n_estimators=120,max_depth=6,min_samples_split=2,min_samples_leaf=3)

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
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred)
print(a)
#new_y=y_d['SpeedRange'].copy()
#new_y_d=pd.DataFrame(new_y)


# In[43]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('6mar.csv')

X=list[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list[['Class','Mean_speed_kmph']].values
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
gbc=GradientBoostingClassifier(learning_rate=0.1,n_estimators=120,max_depth=6,min_samples_split=2,min_samples_leaf=3)

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
#while i<l:
 #   if speed_check_l[i]>=17.5 and speed_check_l[i]<=22.5 :
  #      if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
   #         y_pred_l[i]=value_check[i]
    #    if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
     #       y_pred_l[i]=value_check[i]
#y_pred_f=        
#print(speed_check_l)
# Generate the confusion matrix and classification report
a=confusion_matrix(new_y_d,y_pred)
print(a)
#new_y=y_d['SpeedRange'].copy()
#new_y_d=pd.DataFrame(new_y)


# In[42]:


print("Hello")


# In[46]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('6mar.csv')

X=list[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list[['Class','Mean_speed_kmph']].values
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
gbc=GradientBoostingClassifier(learning_rate=0.1,n_estimators=120,max_depth=6,min_samples_split=2,min_samples_leaf=3)

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
    if speed_check_l[i]>=17.5 and speed_check_l[i]<=22.5 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32.5 and speed_check_l[i]<=37.5 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47.5 and speed_check_l[i]<=52.5 :
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

#new_y=y_d['SpeedRange'].copy()
#new_y_d=pd.DataFrame(new_y)


# In[48]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('6mar.csv')

X=list[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list[['Class','Mean_speed_kmph']].values
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
gbc=GradientBoostingClassifier(learning_rate=0.1,n_estimators=120,max_depth=6,min_samples_split=2,min_samples_leaf=3)

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
    if speed_check_l[i]>=17.5 and speed_check_l[i]<=22.5 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Slow':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Slow' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=32.5 and speed_check_l[i]<=37.5 :
        if value_check[i]=='Normal' and y_pred_l[i]=='Fast':
            y_pred_l[i]=value_check[i]
        if value_check[i]=='Fast' and y_pred_l[i]=='Normal':
            y_pred_l[i]=value_check[i]
    if speed_check_l[i]>=47.5 and speed_check_l[i]<=52.5 :
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
#new_y=y_d['SpeedRange'].copy()
#new_y_d=pd.DataFrame(new_y)


# In[56]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('6mar.csv')

X=list[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list[['Class','Mean_speed_kmph']].values
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
gbc=GradientBoostingClassifier(learning_rate=0.125,n_estimators=125,max_depth=5,min_samples_split=5,min_samples_leaf=6)

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
#new_y=y_d['SpeedRange'].copy()
#new_y_d=pd.DataFrame(new_y)


# In[55]:


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

list=pd.read_csv('6mar.csv')

X=list[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list[['Class','Mean_speed_kmph']].values
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
#new_y=y_d['SpeedRange'].copy()
#new_y_d=pd.DataFrame(new_y)
#model = XGBClassifier(eta=0.5,max_depth=5,seed=21,sub_sample=0.5,nthread=3)


# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
df=pd.read_csv('6mar.csv')

honk_l=df[['Honk_duration','Mean_speed_kmph','Timelevel','Zone',]].values
honk_d=pd.DataFrame(honk_l)

honk_d.loc[honk_d[0]<=4,'Honk_bin']='1'
honk_d.loc[(honk_d[0]>4) &(honk_d[0]<=12),'Honk_bin']='2'
honk_d.loc[(honk_d[0]>12) &(honk_d[0]<=20),'Honk_bin']='3'
honk_d.loc[(honk_d[0]>20) &(honk_d[0]<=28),'Honk_bin']='4'
honk_d.loc[(honk_d[0]>28),'Honk_bin']='5'
honk_d_f=pd.DataFrame(honk_d)

l=len(honk_d)
print(l)

i=0

while i<l:
    if (honk_d.iloc[i,2]==1) & (honk_d.iloc[i,3]=='market') & (honk_d.iloc[i,4]=='1') & (honk_d.iloc[i,1]>48):
        df=df.drop(i)
        print("A")
    if (honk_d.iloc[i,2]==1) & (honk_d.iloc[i,3]=='market') & (honk_d.iloc[i,4]=='1') & (honk_d.iloc[i,1]<16):
        df=df.drop(i)
        print("A")
    if (honk_d.iloc[i,2]==1) & (honk_d.iloc[i,3]=='normal_city') & (honk_d.iloc[i,4]=='1') & (honk_d.iloc[i,1]>58):
        df=df.drop(i)
        print("b")
   
    if (honk_d.iloc[i,2]==2) & (honk_d.iloc[i,3]=='market') & (honk_d.iloc[i,4]=='1') & (honk_d.iloc[i,1]>48) :
        df=df.drop(i)
        print("c")
    if (honk_d.iloc[i,2]==2) & (honk_d.iloc[i,3]=='market') & (honk_d.iloc[i,4]=='1') & (honk_d.iloc[i,1]<15) :
        df=df.drop(i)
        print("c")
        #print("A")
    if (honk_d.iloc[i,2]==2) & (honk_d.iloc[i,3]=='normal_city') & (honk_d.iloc[i,4]=='1') & (honk_d.iloc[i,1]>52):
        df=df.drop(i)
        print("d")
    if (honk_d.iloc[i,2]==2) & (honk_d.iloc[i,3]=='normal_city') & (honk_d.iloc[i,4]=='1') & (honk_d.iloc[i,1]<18):
        df=df.drop(i)
        print("d")
    if (honk_d.iloc[i,2]==3) & (honk_d.iloc[i,3]=='market') & (honk_d.iloc[i,4]=='1') & (honk_d.iloc[i,1]>55) :
        df=df.drop(i)
        print("e")
    
    if (honk_d.iloc[i,2]==4) & (honk_d.iloc[i,3]=='highway') & (honk_d.iloc[i,4]=='1') & (honk_d.iloc[i,1]>72) :
        df=df.drop(i)
        print("g")
    if (honk_d.iloc[i,2]==4) & (honk_d.iloc[i,3]=='market') & (honk_d.iloc[i,4]=='1') & (honk_d.iloc[i,1]>48) :
        df=df.drop(i)
        print("h")
    if (honk_d.iloc[i,2]==4) & (honk_d.iloc[i,3]=='normal_city') & (honk_d.iloc[i,4]=='1') & (honk_d.iloc[i,1]>60):
        df=df.drop(i)
        print("i")
    
    i=i+1
    #print(i)
print(len(df))    
#list=pd.read_csv('6mar.csv')

X=df[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=df[['Class','Mean_speed_kmph']].values
y_d=pd.DataFrame(y)
print(len(y_d))
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




