import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
data=pd.read_csv('Paytm.csv')
print(data.columns)
print(data.isna().sum())
print(data.describe())
print(data.info())
for i in data.select_dtypes(include='number').columns.values:
    sn.boxplot(data[i])
    plt.show()

lab=LabelEncoder()
for i in data.select_dtypes(include='object').columns.values:
    data[i]=lab.fit_transform(data[i])

for i in data.columns.values:
    isna=data[i].isna().sum().tolist()
    if isna > 0:
        data[i]=data[i].fillna(data[i].mean())

for i in data.columns.values:
        print(data[i].value_counts())
        print(data[i].value_counts().index)

for i in  data.columns.values:
        if len(data[i].value_counts())<=5:
            sn.countplot(data[i])
            plt.show()


for i in data.columns.values:
    if len(data[i].value_counts()) <= 5:
        for j in data[i].value_counts().index.values:
            print('------------------------------------------')
            print(f"The information about the column {i}")
            val=data[data[i]==j]
            for k in val.select_dtypes(include='object').columns.values:
                index = val[k].value_counts().index.values
                value = val[k].value_counts().values
                if (len(index) and len(value)) <= 10:
                    plt.pie(value, labels=index, autopct='%1.1f%%')
                    plt.title(f'the values and their counts related to  {i} column')
                    plt.legend()
                    plt.show()

plt.figure(figsize=(17, 6))
corr = data.corr(method='spearman')
my_m = np.triu(corr)
sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
plt.show()

for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        sn.distplot(data[i], label=f"{i}", color='red')
        sn.distplot(data[j], label=f"{j}", color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()

for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        sn.histplot(data[i], label=f"{i}", color='red')
        sn.histplot(data[j], label=f"{j}", color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()