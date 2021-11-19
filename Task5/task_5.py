import pandas as pd
import matplotlib.pyplot as plt
import re

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
print(data)
message_per_date = data.groupby('date').size()
print(message_per_date)
message_per_date.plot()
plt.gcf().autofmt_xdate()
plt.show()
