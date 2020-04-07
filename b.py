import requests
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

data_path = "/home/ambuj/Desktop/integration-plugins/ml_plugins/COVID19/data.dat"
hos_data_path = "/home/ambuj/Desktop/integration-plugins/ml_plugins/COVID19/total_beds.csv"

def date_ex(x):
    last_date = x[-1]
    x_new = x[:]
    m,d,y = last_date.split('/')
    for i in range(60):
        d_new = int(d)+i+1
        m_new = int(m)
        if d_new > 30:
            d_new -= 30
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
date_list_new = list(date_map.values())
date_list_new.sort()
country_names = list(set(country_names))
data_dict = dict((el,{}) for el in country_names)
for key in data_dict:
    data_dict[key] = dict((el,{}) for el in ['confirmed', 'deaths', 'recovered'])
    data_dict[key]['confirmed'] = dict((el,0) for el in date_list)
    data_dict[key]['deaths'] = dict((el,0) for el in date_list)
    data_dict[key]['recovered'] = dict((el,0) for el in date_list)
print("Data sorted...")
for i in range(len(res['confirmed']['locations'])):
    for date in res['confirmed']['locations'][i]['history']:
        data_dict[res['confirmed']['locations'][i]['country']]['confirmed'][date] = res['confirmed']['locations'][i]['history'][date]
for i in range(len(res['deaths']['locations'])):
    for date in res['deaths']['locations'][i]['history']:
        data_dict[res['deaths']['locations'][i]['country']]['deaths'][date] = res['deaths']['locations'][i]['history'][date]
for i in range(len(res['recovered']['locations'])):
    for date in res['recovered']['locations'][i]['history']:
        data_dict[res['recovered']['locations'][i]['country']]['recovered'][date] = res['recovered']['locations'][i]['history'][date]
#pp = PdfPages('country_limits.pdf')
fig, ax = plt.subplots()
with PdfPages('country_limits.pdf') as pdf:
    for country in country_names:
        data = dict((el,0) for el in date_list_new)
        for date in data_dict[country]['confirmed']:
            #print(date, data_dict[country]['confirmed'][date], data_dict[country]['deaths'][date], data_dict[country]['recovered'][date])
            data[date_map[date]] += data_dict[country]['confirmed'][date] - (data_dict[country]['deaths'][date] + data_dict[country]['recovered'][date])
        y = []
        x = []
        f = 0
        dates = list(data.keys())
        dates.sort()
        x_dates = []
        if data[dates[-1]]>500:
            for key in dates:
                if data[key] or f:
                    y.append(data[key])
                    x.append(f)
                    x_dates.append(key)
                    f += 1
            total_beds = pd.read_csv(hos_data_path)
            x_predict_dates = date_ex(x_dates)
            x_predict = list(range(len(x)+60))
            x_array = np.reshape(x, (-1, 1))
            x_predict_array = np.reshape(x_predict, (-1, 1))
            poly = PolynomialFeatures(degree=4)
            X_ = poly.fit_transform(x_array)
            predict_ = poly.fit_transform(x_predict_array)
            clf = linear_model.LinearRegression()
            clf.fit(X_, y)
            y_predict = clf.predict(predict_)
            beds = 0
            if country in list(total_beds['Country']):
                for i, cnt in enumerate(total_beds['Country']):
                    if cnt == country:
                        beds = total_beds['total bed'][i]
                #trained_data[country] = {'x': x_predict_dates, 'y': list(y_predict)}
            if beds:
                print(country)
                f = 2
                for i in range(1, len(y_predict)):
                    if y_predict[-i] > 0:
                        break
                legends = []
                ax.plot(x, y)
                ax.plot(x_predict_dates[:-i], y_predict[:-i])
                ax.plot(x_predict_dates[:-i], [beds]*len(x_predict_dates[:-i]))
                ax.legend(['Actual', 'Predicted', 'Capacity='+str(beds)], loc='upper right')
                plt.xticks(rotation=45)
                for label in ax.get_xaxis().get_ticklabels()[::]:
                    label.set_visible(False)
                for label in ax.get_xaxis().get_ticklabels()[::4]:
                    label.set_visible(True)
                ax.set_title(country)
                pdf.savefig()
                plt.cla()
#pp.close()
print("Predictions done...")
'''
            #q = 0
            #der = 0.0 
            #for i in range(len(y)-1):
            #     if (y[i+1]-y[i]):
            #         der += (y[i+1] - y[i])
            #        q +=1
            #der = der/q
            #spread_data[country] = der
'' ##scatter v density scatter##
x = []
y = []
c = []
for key in spread_data:
    if key in pop_data and key != 'Singapore':
        x.append(pop_data[key])
        y.append(spread_data[key])
        c.append(key)
plt.scatter(x,y)
for i,z in enumerate(c):
    plt.annotate(z ,(x[i], y[i]))
plt.xlabel("Individuals/km")
plt.ylabel("Virus spread rate")
plt.show()
'''