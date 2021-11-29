import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr, spearmanr, kendalltau
import seaborn as sbn
from scipy.stats import norm
from statsmodels.graphics.gofplots import qqplot
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

path = "your_qgis_output.csv"

data = pd.read_csv(path, sep = ",")

data.columns = ['trackpoint', 'ufp', 'street_class', 'kfz', 'lkw', 'tot_traffic']
data.set_index("trackpoint", inplace=True)



where_nan = data['ufp'].isna()
data = data[~where_nan]

traffic_subset = data[~data['tot_traffic'].isna()]
data['street_class'].fillna('No_street', inplace=True)

##
corr_traff = pearsonr(traffic_subset['traffic_norm'], traffic_subset['ufp'])    # Pearson nimmt normalverteilte Daten an!
print("------ Pearson's Korrelationskoeffizient -----")
print('r = %.3f      p = %.3E' % (corr_traff[0], corr_traff[1]))
print("\n")

# Alles numerischen Daten kreuzkorrelieren:
data.corr()
data.iloc[:, 2:-1].corr()


## Verteilungen

# Wir sehen aber, dass die Daten nicht normalverteilt sind.
sbn.distplot(data['ufp'], kde=False, color='blue', bins=100)
plt.title('Absolute Häufigkeitsverteilung der UFP-Konzentration', fontsize=18)
plt.xlabel('Konzentration [#/cm^3]', fontsize=16)
plt.ylabel('Häufigkeitsdichte', fontsize=16)
plt.show()

sbn.distplot(data['ufp'], color='blue', bins=100)
plt.title('Relative Häufigkeitsverteilung der UFP-Konzentration', fontsize=18)
plt.xlabel('Konzentration [#/cm^3]', fontsize=16)
plt.ylabel('Relative Häufigkeitsdichte', fontsize=16)
mu, std = norm.fit(data['ufp'])
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.show()

qqplot(data['ufp'], line='s')
plt.show()

# Auch die Verkehrszählungen sind fern von Normalverteilung
sbn.distplot(data['kfz'], kde=False, color='blue', bins=100)
plt.show()

sbn.distplot(data['lkw'], kde=False, color='orange', bins=100)
plt.show()

# Weitere Möglichkeiten: Schiefe berechnen --> scipy.stats.skew()
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html
# D'Agostino test: scipy.stats.normaltest() oder viele weitere Tests.
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html

## Verteilung hinsichtlich der Straßenklassen

street_classes = ["No_street", "I", "II", "III", "IV", "V"]
data['street_class'] = pd.Categorical(data['street_class'], categories= street_classes)
data_ordered = data.sort_values(by="street_class")

sbn.stripplot('street_class', 'ufp', data=data_ordered)
plt.title('Straßenklasse vs Ultrafeinstaubkonzentration', fontsize=18)
plt.ylabel('Konzentration [#/cm^3]', fontsize=12)
plt.xlabel('')
plt.show()

sbn.violinplot('street_class', 'ufp', data=data_ordered, scale="count")
plt.title('Straßenklasse vs Ultrafeinstaubkonzentration', fontsize=18)
plt.ylabel('Konzentration [#/cm^3]', fontsize=12)
plt.xlabel('')
plt.show()

## Verteilung hinsichtlich des Verkehrsaufkommens

sbn.scatterplot('kfz', 'ufp', data=data)
sbn.scatterplot('lkw', 'ufp', data=data, marker="+")
plt.legend(labels=["KfZ", "LKW"])
plt.title('Verkehrsmengen vs UFP-Konzentration', fontsize=14)
plt.ylabel('Konzentration [#/cm^3]', fontsize=12)
plt.xlabel('')
plt.show()

sbn.scatterplot('lkw', 'ufp', data=data)
plt.title('Verkehrsmengen (LKW) vs UFP-Konzentration', fontsize=14)
plt.ylabel('Relative Konzentration [#/cm^3]', fontsize=12)
plt.xlabel('')
plt.show()


## Überprüfen von nicht-linearen Korrelationen:


def corr_check(x, y, significance=0.05, method="spearman"):
    if method == "spearman":
        coeff, pvalue = spearmanr(x, y)
        print("------ Spearman's Rangkorrelationskoeffizient (rho) -----")
        print('ρ = %.3f' % coeff)
    elif method == "kendalltau":
        coeff, pvalue = kendalltau(x, y)
        print("------ Kendall's Rangkorrelationskoeffizient (tau) -----")
        print('τ = %.3f' % coeff)

    if pvalue > significance:
        print('Stichproben korrelieren nicht mit p=%.3E' % pvalue)
    else:
        print('Stichproben korrelieren mit p=%.3E' % pvalue)
    print("\n")


corr_check(traffic_subset['tot_traffic'], traffic_subset['ufp'])   # Spearmans rank correlation (lineare und nicht-lineare Zusammenhänge, sensitiver gegenüber Fehlern/Diskrepanzen in den Daten)
corr_check(traffic_subset['tot_traffic'], traffic_subset['ufp'], method="kendalltau")    # Kendall's tau correlation (unempfindlicher)

##
corr_check(data['street_class'], data['ufp'])
corr_check(data['street_class'], data['ufp'], method='kendalltau')


## Univariate lineare Regression:

predictor = 'kfz'
X = traffic_subset[[predictor]].to_numpy()
Y = traffic_subset[['ufp']].to_numpy()
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.xlabel(str('Mittleres Verkehrsaufkommen' + ' 100m um den Messpunkt'))
plt.ylabel('UFP-Konzentration [#/cm^3]')
plt.show()

plt.scatter(Y, Y_pred)
plt.xlabel('Gemessene UFP-Konzentration [#/cm^3]')
plt.ylabel('Modellierte UFP-Konzentration [#/cm^3]')
plt.show()

print('------ Lineare Regression -----')
print('Funktion: y = %.3f * x + %.3f' % (linear_regressor.coef_[0], linear_regressor.intercept_))
print("R² Score: {:.2f}".format(linear_regressor.score(X, Y)))
print("\n")

## Multivariate lineare Regression (nur zu Demonstrationszwecken! Massive Multikolinearität, da Prädiktoren stark korrelieren!):

predictor1 = 'tot_traffic'
predictor2 = 'lkw'
feature = traffic_subset[[predictor1, predictor2]].to_numpy()
result = traffic_subset[['ufp']].to_numpy()
linear_regressor = LinearRegression()
linear_regressor.fit(feature, result)
result_pred = linear_regressor.predict(feature)

plt.scatter(result, result_pred)
plt.xlabel('Gemessene UFP-Konzentration [#/cm^3]')
plt.ylabel('Modellierte UFP-Konzentration [#/cm^3]')
plt.show()

print('------ Lineare Regression -----')
print('Funktion: y = %.3f * x1 + %.3f * x2 + %.3f' % (linear_regressor.coef_[0][0], linear_regressor.coef_[0][1], linear_regressor.intercept_))
print("R² Score: {:.2f}".format(linear_regressor.score(feature, result)))
print("\n")

## Multivariate Regression mit ordinalen Prädiktoren

t = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(), ['street_class']),
    ('scale', StandardScaler(), ['tot_traffic']),
])

features = t.fit_transform(traffic_subset)
result = traffic_subset['ufp']

# Train the linear regression model
linear_regressor = LinearRegression()
model = linear_regressor.fit(features, result)

result_pred = linear_regressor.predict(features)

plt.scatter(result_pred, result)
plt.xlabel('Modellierte UFP-Konzentration [#/cm^3]')
plt.ylabel('Gemessene UFP-Konzentration [#/cm^3]')
plt.title('Lin. Regression Straßenklasse / Verkehrsaufkommen', fontsize=14)
plt.show()

print('------ Lineare Regression -----')
print('Funktion: y = %.3f * x1 + %.3f * x2 + %.3f + %.3f * x3 + %.3f * x4 + %.3f * x5 + %.3f * x6' %
      (linear_regressor.coef_[0], linear_regressor.coef_[1], linear_regressor.coef_[2],
       linear_regressor.coef_[3], linear_regressor.coef_[4], linear_regressor.coef_[5], linear_regressor.intercept_))
print("R² Score: {:.2f}".format(linear_regressor.score(features, result)))
print("\n")


## Multivariate Random Forest:
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=0)
model = rf_reg.fit(features, result)

Y_pred = rf_reg.predict(features)

plt.scatter(Y_pred, result)
plt.xlabel('Modellierte UFP-Konzentration [#/cm^3]')
plt.ylabel('Gemessene UFP-Konzentration [#/cm^3]')
plt.title('RF-Regression Straßenklasse / Verkehrsaufkommen', fontsize=14)
plt.show()

print('------ Random Forest Regression -----')
print("R² Score: {:.2f}".format(rf_reg.score(features, result)))
print("\n")



