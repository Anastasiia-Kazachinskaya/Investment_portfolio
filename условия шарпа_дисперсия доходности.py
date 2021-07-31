import pandas as pd
import math
from sympy import *
import numpy as np
from scipy.optimize import minimize
from sympy import *

from scipy.optimize import minimize


securities = ['Россия20', 'AAPL', 'ADBE', 'AFKS', 'AKRN', 'AMZN', 'BABA', 'CSCO', "CMC", "CFX", 'ETSY', 'FORD', 'FRHC',
              'FTCH', 'GOOGL', 'GTHX', 'JNJ', 'MOMO', 'MNST', 'MSFT', 'NOKIA', 'OSUR', 'OZON', 'PFE', 'REGI', 'SPCE',
              'SHOP', 'SBER_p', 'TSLA', "V", 'VOD', 'YNDX']  # наши ценные бумаги
print(securities)
print(len(securities))
portfolio = dict()  # наш портфель
securities.remove('AKRN')
securities.remove('NOKIA')
index = pd.read_csv(r'/Users/alexeychurgel/pythonProject/данные по акциям/Прошлые данные - DJIA.csv')
index['Изм. %'] = index['Изм. %'].apply(lambda x: float(x.replace(',', '.')[:len(x) - 1]))
index_mean = np.mean(index['Изм. %'])  # считаем среднее арифметическое по статистическим данным
index_dispersion = np.var(index['Изм. %'], ddof=1, axis=0)  # считаем дисперсию (несмещенную) по статистическим данным
path = '/Users/alexeychurgel/pythonProject/данные по акциям'
print(securities)
print(len(securities))
for paper in securities:
    data = pd.read_csv(f'{path}/Прошлые данные - {paper}.csv')
    data = data[data['Дата'].isin(index['Дата'])].reset_index(drop=True)# удаляем данные за те дни, которых нет в index
    data['Изм. %'] = data['Изм. %'].apply(lambda x: float(x.replace(',', '.')[:len(x) - 1]))
    mean = np.mean(data['Изм. %'])  # считаем среднее арифметическое доходности актива по статистическим данным
    ind = index[index['Дата'].isin(data['Дата'])].reset_index(drop=True)
    index_cov = np.cov(ind['Изм. %'], data['Изм. %'])[0, 1]
    betta = index_cov / index_dispersion
    alpha = mean - betta * index_mean
    ksi_mean = np.mean(data['Изм. %'] - alpha - betta * ind['Изм. %'])  # 1 необходимое условие
    ksi_disp = (1 / (len(index) - 2)) * ((data['Изм. %'] - alpha - betta * ind['Изм. %']) ** 2).sum()  # оценки дисперсии # 𝜓i^2 случайной величины 𝜉𝑖t
    ksi = data['Изм. %'] - alpha - betta * ind['Изм. %']
    ksi_d = np.var(ksi)
    expected_profitability = round(alpha + betta * index_mean,4)
    portfolio[paper] = {'data': data['Изм. %'], 'ind': ind, 'mean': mean, 'index_cov': index_cov,
                        'expected_profitability': expected_profitability, 'ksi': ksi, 'ksi_mean': ksi_mean,
                        'ksi_disp': ksi_disp, 'ksi_d': ksi_d, 'alpha': alpha, 'betta': betta}
print(securities)
print(len(securities))
print("Условие 1")
print(securities)
print(len(securities))
for paper in securities:
    print(f"{paper}: {str(abs(portfolio[paper]['ksi_mean']))[:5]}")
securities = [item for item in securities if int(str(abs(portfolio[item]['ksi_mean']))[:1]) < 2]
print("Условие 3")
print(securities)
print(len(securities))
for paper in securities:
    print(f"{paper}: {round(portfolio[paper]['ksi_d'] - portfolio[paper]['ksi_disp'],4)}")
securities = [item for item in securities if portfolio[item]['ksi_d'] - portfolio[item]['ksi_disp'] < 0.5]
print("Условие 4")
print(securities)
print(len(securities))
for i in range(len(securities)):
    print(f"{securities[i]}: {round(abs(portfolio[securities[i]]['mean'] * index_mean),4)}")
print("Условие 2")
print(securities)
print(len(securities))
for i in range(len(securities)):
    j = 0
    for pap in dict([(key, value) for key, value in portfolio.items() if key in securities[i + 1:]]): # $#119894; ≠ j
        j += 1
        summ = 0
        var1 = 0
        for k in portfolio[securities[i]]['ksi']:
            var2 = 0
            for kk in portfolio[pap]['ksi']:
                if var1 != var2: # $#119905; ≠ t
                    summ += k * kk
                var2 +=1
            var1+=1
        print(f"{securities[i]} & {securities[i + j]}: {round(summ / ((len(portfolio[securities[i]]['ksi']) - 1) * (len(portfolio[pap]['ksi']) - 1) - 2),4)}")
expected_prof_matrix = np.array([])
print(securities)
print(len(securities))
print('Oжидаемые доходности активов:')
for i in range(len(securities)):
    exp_p = portfolio[securities[i]]['expected_profitability']
    print(f"{securities[i]}: {exp_p}")
expected_prof_matrix = np.hstack((expected_prof_matrix, exp_p))
expected_prof_matrix = expected_prof_matrix.reshape(-1, 1)
print('Матрица ковариаций доходностей активов:')
cov_matrix = np.array([])
print(securities)
print(len(securities))
for paper1 in securities:
    if paper1 != "Россия20":
        b1 = portfolio[paper1]['betta']
        for paper2 in securities:
            if paper2 != "Россия20":
                b2 = portfolio[paper2]['betta']
                cov_matrix = np.hstack((cov_matrix, round(b1 * b2 * index_dispersion,2)))
cov_matrix = cov_matrix.reshape(len(securities) -1, -1)
print(cov_matrix)
print(cov_matrix.shape)

x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,p,q,w=symbols(' x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 p q w' , float= True)
v1=Matrix([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11])
v2=v1.T
w=0
for i in np.arange(0,10):
         for j in np.arange(0,10):
             w=w+v1[p.subs({p:i}),q.subs({q:0})]*v2[p.subs({p:0}),q.subs({q:j})]*cov_matrix[p.subs({p:i}),q.subs({q:j})]
print("Дисперсия доходности портфеля (функция риска):\n%s"%w)


