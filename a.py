import requests
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

data_path = "/home/ambuj/Desktop/integration-plugins/ml_plugins/COVID19/data.dat"

def date_ex(x):
    last_date = x[-1]
    x_new = x[:]
    m,d,y = last_date.split('/')
    for i in range(14):
        d_new = int(d)+i+1
        m_new = int(m)
        if d_new > 31:
            d_new -= 31
            m_new += 1
        if len(str(d_new))%2:
            d_new = "0" + str(d_new)
        else:
            d_new = str(d_new)
        if len(str(m_new))%2:
            m_new = "0" + str(m_new)
        else:
            m_new = str(m_new)
        x_new.append(m_new +"/"+ d_new +"/"+ y)
    return x_new

country_names = []
date_list = []
response = requests.get('https://coronavirus-tracker-api.herokuapp.com/all')
res = response.json()
print("Data read...")
for i in range(len(res['confirmed']['locations'])):
    date_list.extend(res['confirmed']['locations'][i]['history'].keys())
    country_names.append(res['confirmed']['locations'][i]['country'])
date_list = list(set(date_list))
date_map = {}
for date in date_list:
    m,d,y = date.split('/')
    if len(m)%2:
        m = '0'+m
    if len(d)%2:
        d = '0'+d
    date_map[date] = m+"/"+d+"/"+y
date_list = list(date_map.values())
date_list.sort()
country_names = list(set(country_names))
print("Data sorted...")
c = 0
trained_data = {}
for country in country_names:
    try:
        c += 1
        print(str(c) +" "+ country)
        target_country = country
        data = dict((el,0) for el in date_list)
        for i in range(len(res['confirmed']['locations'])):
            if target_country == res['confirmed']['locations'][i]['country']:
                for date in res['confirmed']['locations'][i]['history']:
                    data[date_map[date]] += res['confirmed']['locations'][i]['history'][date]
        y = []
        x = []
        f = 0
        dates = list(data.keys())
        dates.sort()
        x_dates = []
        for key in dates:
            if data[key] or f:
                y.append(data[key])
                x.append(f)
                x_dates.append(key)
                f += 1
        x_predict_dates = date_ex(x_dates)
        x_predict = list(range(len(x)+14))
        x_array = np.reshape(x, (-1, 1))
        x_predict_array = np.reshape(x_predict, (-1, 1))
        poly = PolynomialFeatures(degree=4)
        X_ = poly.fit_transform(x_array)
        predict_ = poly.fit_transform(x_predict_array)
        clf = linear_model.LinearRegression()
        clf.fit(X_, y)
        y_predict = clf.predict(predict_)
        trained_data[country] = {'x': x_predict_dates, 'y': list(y_predict)}
        legends = []
        plt.plot(x_dates, y)
        plt.plot(x_predict_dates, y_predict)
        plt.legend(['Actual', 'Predicted'], loc='upper right')
        plt.show()
    except Exception:
        pass
print("Predictions done...")
with open(data_path, "w") as f:
    pickle.dump(trained_data, f, 2)
print("Models updated.")
