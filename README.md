# Ex-06-Feature-Transformation
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.
# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
# ALGORITHM
## STEP 1
Read the given Data
## STEP 2
Clean the Data Set using Data Cleaning Process
## STEP 3
Apply Feature Transformation techniques to all the features of the data set
## STEP 4
Save the data to the file
# CODE
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

df = pd.read_csv("/content/Data_to_Transform.csv")
df

df.head()

df.isnull().sum()

df.info()

df.describe()

df1 = df.copy()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
```
# OUPUT
## Dataset:
### ![image](https://user-images.githubusercontent.com/128350225/232397668-aa35d6eb-052d-4d36-8820-cf6f24462335.png)
## Head:
### ![image](https://user-images.githubusercontent.com/128350225/232397736-d65b1426-3bc8-47ab-8671-63d5b03280db.png)
## Null data:
### ![image](https://user-images.githubusercontent.com/128350225/232397792-01cd54ba-236d-41a5-8efb-492efc02fa25.png)
## Information:
### ![image](https://user-images.githubusercontent.com/128350225/232397845-0d5a77cc-4444-40b5-9612-97382fdcaa3b.png)
## Description:
### ![image](https://user-images.githubusercontent.com/128350225/232397906-f3d7074c-dedf-4366-a883-b53d5525ab82.png)
## Highly Positive Skew:
### ![image](https://user-images.githubusercontent.com/128350225/232397964-037030ff-8ebb-4ffc-a58c-1c23b05e6651.png)
## Highly Negative Skew:
### ![image](https://user-images.githubusercontent.com/128350225/232398297-20e2a30b-8a88-4f64-ac01-d2cd142866f5.png)
## Moderate Positive Skew:
### ![image](https://user-images.githubusercontent.com/128350225/232398377-d3f346be-bff9-4dc6-ba93-cca2f48f38a0.png)
## Moderate Negative Skew:
### ![image](https://user-images.githubusercontent.com/128350225/232398495-3e728248-e45f-47ca-bd16-3e94919fb98e.png)
## Log of Highly Positive Skew:
### ![image](https://user-images.githubusercontent.com/128350225/232398566-cc30bb3d-1e76-40c1-82f6-40290118f1f3.png)
## Log of Moderate Positive Skew:
### ![image](https://user-images.githubusercontent.com/128350225/232398642-b0f73d0f-1542-4e39-ab46-f49d0d48d5b1.png)
## Square root tranformation:
### ![image](https://user-images.githubusercontent.com/128350225/232398721-33906fa5-0641-41f5-a2cb-14661176114e.png)
## Power transformation of Moderate Positive Skew:
### ![image](https://user-images.githubusercontent.com/128350225/232398802-d4e213a1-71d2-4d1c-9483-3b34720dbd90.png)
## Power transformation of Moderate Negative Skew:
### ![image](https://user-images.githubusercontent.com/128350225/232398906-c4d579c7-b90e-4e19-a6f1-7f746a9177c6.png)
## Quantile transformation:
### ![image](https://user-images.githubusercontent.com/128350225/232399067-c3dcb197-130a-4fca-8626-73d1c7805279.png)
# Result:
Thus, Feature transformation is performed and executed successfully for the given dataset
