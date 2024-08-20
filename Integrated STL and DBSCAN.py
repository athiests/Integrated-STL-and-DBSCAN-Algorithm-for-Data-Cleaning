import copy
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import solve
from statsmodels.tsa.seasonal import STL
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from deap import base, creator, tools, algorithms

data=pd.read_csv(r'DO.csv',encoding='gbk')
df = pd.DataFrame({'Date': data['time'], 'Value': data['DO']})
df.set_index('Date', inplace=True)

def reverse_nearest_neighbors(X, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    indices = indices[:, 1:]
    reverse_nn = {i: [] for i in range(len(X))}
    for i in range(len(X)):
        for j in indices[i]:
            if i in indices[j]:
                reverse_nn[j].append(i)
    return reverse_nn

def lwlr(X, y, x0, tau=1.0, lambda_reg=1e-5):
    m = X.shape[0]
    X_aug = np.hstack((np.ones((m, 1)), X.reshape(-1, 1), X.reshape(-1, 1) ** 2))
    x0_aug = np.array([1, x0, x0 ** 2])
    W = np.eye(m)
    for i in range(m):
        diff = X_aug[i] - x0_aug
        W[i, i] = np.exp(-diff @ diff.T / (2 * tau ** 2))
    XTWX = X_aug.T @ W @ X_aug
    XTWy = X_aug.T @ W @ y
    XTWX += lambda_reg * np.eye(XTWX.shape[0])
    theta = solve(XTWX, XTWy)
    return x0_aug @ theta

def rnn_loess(X, y, k=5, tau=1.0):
    m = len(X)
    y_smooth = np.zeros_like(y)
    rnn = reverse_nearest_neighbors(X.reshape(-1, 1), k)
    for i in range(m):
        if len(rnn[i]) > 0:
            X_rnn = X[rnn[i]]
            y_rnn = y[rnn[i]]
            y_smooth[i] = lwlr(X_rnn, y_rnn, X[i], tau)
        else:
            y_smooth[i] = y[i]
    return y_smooth

def iterative_trend_smoothing(data, period, k=5, tau=10.0, tolerance=1e-4, max_iter=10):
    X = np.arange(len(data))
    trend = np.zeros_like(data)
    residuals = data.copy()
    result = sm.tsa.seasonal_decompose(data, period=period)

    for _ in range(max_iter):
        new_trend = rnn_loess(X, residuals, k=k, tau=tau)
        if np.max(np.abs(new_trend - result.trend)) < tolerance:
            print("Converged after iteration.")
            break

        trend = result.trend-new_trend
    seasonal = result.seasonal
    residual = data - trend - seasonal
    return trend, seasonal, residual

def slrj(A,a,B,b,residual,trend,seasonal,data_ori):
    can = pd.DataFrame(data=residual)
    miu = np.mean(can.dropna())
    sigma = np.abs(np.std(can.dropna(), ddof=1))
    data_ac=copy.deepcopy(data_ori)
    i = 0
    while i < len(data_ac) - 1:
        if residual[i] < float(miu - B * sigma) or residual[i] > float(miu + A * sigma):
            for k in range(i - 12, i + 12):
                if residual[k] < float(miu - b * sigma) or residual[k] > float(miu + a * sigma):
                    data_ac[k] = trend[k] + seasonal[k]
        i = i + 1
    return(data_ac)

def compute_rnn(X, n_neighbors=5):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    distances, indices = nbrs.kneighbors(X)
    rnn_counts = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        for j in range(1, n_neighbors):
            rnn_counts[indices[i, j]] += 1

    return rnn_counts

def filter_rnn_samples(X, rnn_counts, threshold=5):
    return X[rnn_counts >= threshold]

def dbscan_fitness(individual):
    eps, min_samples = individual
    min_samples = max(1, int(min_samples))
    rnn_counts = compute_rnn(X)
    filtered_X = filter_rnn_samples(X, rnn_counts)
    if len(filtered_X) == 0:
        return -1,
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(filtered_X)
    labels = db.labels_
    non_noise_mask = labels != -1
    unique_labels = set(labels[non_noise_mask])
    if len(unique_labels) < 2:
        return -1,
    score = silhouette_score(filtered_X[non_noise_mask], labels[non_noise_mask])
    return score,

def check_individual(individual):
    eps, min_samples = individual
    if eps <= 0:
        individual[0] = np.random.uniform(0.1, 1.0)
    if min_samples < 1:
        individual[1] = np.random.randint(1, 51)
    return individual

period = 36
custom_trend, custom_seasonal, custom_residual = iterative_trend_smoothing(df['Value'].values, period, k=10, tau=2.0)
DAC_1=slrj(4,3,3,2,custom_residual,custom_trend,custom_seasonal,df['Value'].values)
custom_trend_2, custom_seasonal_2, custom_residual_2 = iterative_trend_smoothing(DAC_1, period, k=10, tau=2.0)
DAC_2=slrj(3,2,3,2,custom_residual_2,custom_trend_2,custom_seasonal_2,DAC_1)
data_fc= copy.deepcopy(DAC_2)
rd = sm.tsa.seasonal_decompose(data_fc, period=36)
lenk=10000
X = pd.DataFrame(rd.seasonal[int(period/2):lenk], columns=['seasonal'])
X.insert(1, 'resid', rd.resid[int(period/2):lenk])
X = StandardScaler().fit_transform(X)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 0.1, 1.0)
toolbox.register("attr_int", np.random.randint, 1, 51)
toolbox.register("individual", tools.initCycle, creator.Individual,(toolbox.attr_float, toolbox.attr_int), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[0.1, 1], up=[1.0, 50], eta=1.0, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", dbscan_fitness)

population = toolbox.population(n=20)
NGEN = 40
CXPB = 0.7
MUTPB = 0.2

for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)

    for ind in offspring:
        check_individual(ind)

    fits = toolbox.map(toolbox.evaluate, offspring)

    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit

    population = toolbox.select(offspring, k=len(population))

best_individual = tools.selBest(population, k=1)[0]
best_eps, best_min_samples = best_individual

for i in range(0, int(len(DAC_2) / lenk) + 1):
    X=[]
    if i == int(len(DAC_2) / lenk):
        X = pd.DataFrame(rd.seasonal[i * lenk:-int(period/2)], columns=['trend'])
        X.insert(1, 'resid', rd.resid[i * lenk:-int(period/2)])
    elif i==0:
        X = pd.DataFrame(rd.seasonal[int(period/2):(i + 1) * lenk], columns=['trend'])
        X.insert(1, 'resid', rd.resid[int(period/2):(i + 1) * lenk])
    else:
        X = pd.DataFrame(rd.seasonal[i * lenk:(i + 1) * lenk], columns=['trend'])
        X.insert(1, 'resid', rd.resid[i * lenk:(i + 1) * lenk])

    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=best_eps, min_samples=int(best_min_samples)).fit(X)
    labels = db.labels_
    if i==0:
        for j in range(int(period/2),lenk):
            if labels[j-18]==-1:
                DAC_2[j]=rd.trend[j]+rd.seasonal[j]
    else:
        for j in range(0,len(labels)):
            if labels[j]==-1:
                DAC_2[j+i * lenk]=rd.trend[j+i * lenk]+rd.seasonal[j+i * lenk]

pd.DataFrame(DAC_2).to_csv('DO_clean.csv')
