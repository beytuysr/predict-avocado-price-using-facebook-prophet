# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:07:20 2020
avocado prices fbprophet
@author: Beytu
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import random
from fbprophet import Prophet

#2.import dataset
avocado_df=pd.read_csv('avocado.csv')

#3.exploring and visualizing the dataset
ah=avocado_df.head()
at=avocado_df.tail()

avocado_df=avocado_df.sort_values('Date')

plt.figure(figsize=(10,10))
plt.plot(avocado_df['Date'],avocado_df['AveragePrice'])
plt.show()

plt.figure(figsize=[25,12])
sns.countplot(x='region',data=avocado_df)
plt.xticks(rotation=45)
plt.show()

sns.countplot(x='year',data=avocado_df)
plt.show()


#4.make predictions

avocado_prohet_df=avocado_df[['Date','AveragePrice']]
avocado_prohet_final_df=avocado_prohet_df.rename(columns={'Date':'ds','AveragePrice':'y'})
m=Prophet()
m.fit(avocado_prohet_final_df)
future=m.make_future_dataframe(periods=365)
forecast=m.predict(future)
figure=m.plot(forecast,xlabel='Date',ylabel='AveragePrice')
plt.show()

figure2=m.plot_components(forecast)
plt.show()




#part2
avocado_df_sample=avocado_df[avocado_df['region']=='West']
avocado_df_sample=avocado_df_sample.sort_values('Date')

plt.figure(figsize=(10,10))
plt.plot(avocado_df_sample['Date'],avocado_df_sample['AveragePrice'])



avocado_prohet_sample_df=avocado_df_sample.rename(columns={'Date':'ds','AveragePrice':'y'})
m=Prophet()
m.fit(avocado_prohet_sample_df)
future=m.make_future_dataframe(periods=365)
forecast=m.predict(future)
figure3=m.plot(forecast,xlabel='Date',ylabel='AveragePrice')
plt.show()

figure4=m.plot_components(forecast)
plt.show()










