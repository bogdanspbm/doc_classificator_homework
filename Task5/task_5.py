import pandas as pd
import matplotlib.pyplot as plt

tmp_data = pd.read_csv('data.csv')
data = pd.DataFrame(tmp_data['date'].str.split('T', 1).tolist(),
                    columns=['date', 'time'])
data['message'] = tmp_data['message']
print(data)
message_per_date = data.groupby('date').size()
print(message_per_date)
message_per_date.plot()
plt.gcf().autofmt_xdate()
plt.show()