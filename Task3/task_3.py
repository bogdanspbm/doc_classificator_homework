import pandas as pd
import matplotlib.pyplot as plt


def task_1():
    df = pd.read_csv('../data/weather.csv', encoding='utf-8', index_col=False, parse_dates=[0])

    df['Year'] = df['Day'].astype(str).str.slice(0, 4)
    y_grouped = df.groupby('Year', as_index=False).mean()

    # print(d_grouped)

    y_min = y_grouped.loc[y_grouped['t'].idxmin()]['Year']
    y_max = y_grouped.loc[y_grouped['t'].idxmax()]['Year']

    print("Min", y_min)
    print("Max", y_max)


def task_2():
    df = pd.read_csv('../data/weather.csv', encoding='utf-8', index_col=False, parse_dates=[0])

    df['Year'] = df['Day'].astype(str).str.slice(0, 4)
    df = df.drop(df[df['t'] <= 0].index)
    y_grouped = df.groupby('Year')['Year'].count()

    y_min = y_grouped.idxmin()
    y_max = y_grouped.idxmax()

    print(y_min)
    print(y_max)


def task_3():
    df = pd.read_csv('../data/weather.csv', encoding='utf-8', index_col=False, parse_dates=[0])

    df['Year'] = df['Day'].astype(str).str.slice(0, 4).astype(int)
    df['Month'] = df['Day'].astype(str).str.slice(5, 7).astype(int)

    df = df.drop(df[df['Month'] < 6].index)
    df = df.drop(df[df['Month'] > 8].index)

    y_grouped = df.groupby('Year', as_index=False).mean()

    y_min = int(y_grouped.loc[y_grouped['t'].idxmax()]['Year'])

    print(y_min)


def task_4():
    df = pd.read_csv('../data/weather.csv', encoding='utf-8', index_col=False, parse_dates=[0])

    df["Delta t"] = abs(df['t'].diff().shift(-1))

    print(df.loc[df['Delta t'].idxmax()])


def task_5():
    df = pd.read_csv('../data/weather.csv', encoding='utf-8', index_col=False, parse_dates=[0])

    df['Year'] = df['Day'].astype(str).str.slice(0, 4)
    y_grouped = df.groupby('Year').mean()

    y_grouped.plot()
    plt.show()


task_2()
