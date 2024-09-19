# EX NO:3-Feature Encoding and Transformation

## AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

## ALGORITHM:

### STEP 1:
Read the given Data.
### STEP 2:
Clean the Data Set using Data Cleaning Process.
### STEP 3:
Apply Feature Encoding for the feature in the data set.
### STEP 4:
Apply Feature Transformation for the feature in the data set.
### STEP 5:
Save the data to the file.

## FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

## Methods Used for Data Transformation:
  ### 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  ### 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

## CODING AND OUTPUT:
### Developed by : SRIVATSAN G
### Reg No : 212223230216

```python

import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```

![image](https://github.com/user-attachments/assets/e0aec733-2a0b-471d-9fa5-0fabb153b7b1)



```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![image](https://github.com/user-attachments/assets/becb3886-8aef-4a3e-b487-e211b97c3aa0)

```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![image](https://github.com/user-attachments/assets/3c6eee96-432b-41e4-bb82-d1f03d385486)

```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/32c72023-eccb-4a20-99c8-7c364b864287)


```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```

![image](https://github.com/user-attachments/assets/debe4f98-9834-4a66-966e-5f89f750d292)


```py
df2=pd.concat([df2,enc],axis=1)
df2
```

![image](https://github.com/user-attachments/assets/71f4cb37-c6d6-4b71-9161-99944a263c70)

```py
pd.get_dummies(df2,columns=["nom_0"])
```

![image](https://github.com/user-attachments/assets/49dbd119-45d5-4e17-90a1-2639e695a23e)



```py
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/38c5b837-3efe-4e99-8a41-bb05f375add6)


```py
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```

![image](https://github.com/user-attachments/assets/cd73d7e5-c3ff-4b74-b71e-9cf5a39cf0f4)


```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

![image](https://github.com/user-attachments/assets/62ff6187-87b7-4755-8865-c849d2b12c83)


```py
dfb=pd.concat([df,nd],axis=1)
dfb
```

![image](https://github.com/user-attachments/assets/6ff65b3b-37b0-4f28-890f-24f0d27d979b)


```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

![image](https://github.com/user-attachments/assets/a1ddebce-d8a0-4536-8328-29131605feed)



```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```

![image](https://github.com/user-attachments/assets/8b81afc8-40e2-45e6-9f24-727274c490b6)


```py
df.skew()
```

![image](https://github.com/user-attachments/assets/882b6b7a-772d-4e44-919f-6c9282931174)


```py
np.log(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/69674890-db2a-47fe-bf9b-526541acc9a0)


```py
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/16c029e0-f558-4b1c-ac0d-d6c9f42afc90)


```py
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/60633172-0989-4e74-9c48-8afecf004f78)


```py
np.square(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/2b0aa005-147e-4d55-8daa-aff5753ed58b)


```py
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![image](https://github.com/user-attachments/assets/4232b20f-6374-485a-b35e-7f464a3befd2)


```py
df.skew()
```
![image](https://github.com/user-attachments/assets/ea7550c7-3e77-405e-8732-703dcfdda357)


```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
```

![image](https://github.com/user-attachments/assets/5f6c97e5-764d-490a-9fd3-b60837c73f42)


```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/96de7554-5bcb-46ff-9559-e2fa0ca6f121)


```py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/41ecccfe-3e61-4bf6-aaa7-8e190be94035)


```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/c9ef138b-f4c9-402a-9e1a-6be18afee13e)

```py
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/daec7a5b-b98c-4e6c-91cd-44bdad62a604)


```py
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/8eb8b216-3fd7-47b0-9ebb-4a796683d655)



## RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
