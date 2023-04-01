import pandas as pd
df=pd.read_csv(r"/home/abhay/Downloads/winequalityN.csv")
df['good wine']=0
df['good wine']=(df['quality']>6).replace([True,False],[1,0])
df.dropna()
df.dropna(inplace=True)
x = df[['fixed acidity' , 'chlorides' , 'pH' , 'alcohol' , 'residual sugar', ]]
y = df['good wine']
from sklearn.naive_bayes import GaussianNB

nve = GaussianNB()
nve.fit(x , y)
import pickle

pickle.dump(nve,open('model.pkl','wb'))
