from operator import le

import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

tmp_data = pd.read_csv('data.csv')
data = pd.DataFrame(tmp_data['date'].str.split('T', 1).tolist(),
                    columns=['date', 'time'])
data['message'] = tmp_data['message']
data = data.dropna()
data = data[data['message'].str.contains('Ж|М|м|ж')]
data = data[data['message'].str.contains('1|2|3|4|5|6|7|8|9|0')]
data['tmp'] = data['message'].str.extract('([МЖмж].[0123456789].)', flags=re.IGNORECASE, expand=False).str.lower()
data['gender'] = data['tmp'].str.extract('([МЖмж])', flags=re.IGNORECASE, expand=False).str.lower()
data['age'] = data['tmp'].str.extract('([0123456789][0123456789])', flags=re.IGNORECASE, expand=False).str.lower()
data = data.dropna()

data = data[data['message'].str.contains('bau|BAU|Bau')]
data['bau'] = data['message'].str.extract('(...[0123456789.,][0123456789][0123456789].bau)', flags=re.IGNORECASE,
                                          expand=False).str.lower()
data['bau'] = data['bau'].str.replace(' bau', '')
data['bau'] = data['bau'].str.replace(',', '.')
data['bau'] = data['bau'].str.replace('.[0123456789][.]', '')
data['bau'] = data['bau'].str.replace('(\D)', '')
data = data[data['bau'].str.len() < 5]
data['bau'] = pd.to_numeric(data['bau'], errors='ignore')
data = data.dropna()

data_for_plot = data.groupby(['age'])['bau'].mean().reset_index()
data_for_plot.plot()
plt.show()
print(data_for_plot)
