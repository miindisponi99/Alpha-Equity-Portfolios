import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn.utils import resample
import seaborn as sns
import statsmodels.api as sm


### Functions
## Find ERC Weights
def ERC(cov):
    def objective(y,cov):
        util=np.dot(np.dot(y.T,cov),y)
        return util
    n=len(cov)
    # Initial conditions: equal weights
    y=np.ones([n])/n                 
    # Weights between 0% and 100%: no shorts
    b_=[(0.,None) for i in range(n)]
    # Constraints
    c_= ({'type':'eq', 'fun': lambda y: sum(np.log(y))-10. })
    optimized=opt.minimize(objective,y,(cov),
              method='SLSQP',bounds=b_,constraints=c_,options={'maxiter': 100, 'ftol': 1e-08})
    ys=optimized.x
    return ys/sum(ys)

## Find Minimum Variance Weights
def MV(cov_matrix):
    def objective(W, C):
        # Calculate mean/variance of the portfolio
        varp=np.dot(np.dot(W.T,cov_matrix),W)
        # Objective: min vol
        util=varp**0.5
        return util
    n=len(cov_matrix)
    # Initial conditions: equal weights
    W=np.ones([n])/n                     
    ## Constraints
    # Positivity constraint
    bounds=[(0.,1.) for i in range(n)]
    # Unitary budget constraint
    def Cons_Budgt(W):
        Cons_Budgt=sum(W)-1
        return Cons_Budgt
    c_= ({'type':'eq', 'fun': Cons_Budgt })
    optimized=opt.minimize(objective,W,(cov_matrix),
                                      method='SLSQP',constraints=c_,bounds=bounds,options={'maxiter': 100, 'ftol': 1e-08})
    return optimized.x


### Setup for the portfolios
data_excel = pd.read_excel('EUROSTOXX50.xlsx', 'RETURNS',usecols="C:AX")
ret = np.array(data_excel)
benchmark = np.array(pd.read_excel('EUROSTOXX50.xlsx', 'RETURNS',usecols="B"))
ret_bm = benchmark[:,0]        
n_assets = ret.shape[1]
l_stocks = data_excel.columns
n_obs = ret.shape[0]
annualized_ret_bm = (1 + np.mean(benchmark))**12 - 1
var_bm = ret_bm.var()
annualized_ret=np.zeros((1,n_assets))
beta = np.zeros((1,n_assets))

# Loading for alpha calculation
for i in range(n_assets):
    annualized_ret[:,i] = (1 + np.mean(ret[:,i]))**12 - 1
    beta[:,i] = np.cov((ret[:,i],ret_bm))[0,1]/var_bm

alpha = np.array(annualized_ret - beta * annualized_ret_bm).T[:,0]
alpha_best = alpha > np.quantile(alpha,0.6875)
alpha_worst = alpha < np.quantile(alpha,0.3125)

### "alpha ERC" equity portfolio
## ERC weights
w_ERC=np.zeros((n_assets))
cov_ERC=np.cov(ret[:,alpha_best],rowvar=0)
w_ERC[alpha_best]= ERC(cov_ERC)
hd_ERC = 1/(np.sum(w_ERC**2))

## Returns ERC portfolio
ret_erc_port = np.sum(np.prod((ret,w_ERC.T)),axis=1)

### "alpha CVaR" portfolio
## Selecting 15 stocks with the highest alpha
ret_select = ret[:,alpha_best]
n_ret_select=ret_select.shape[1]

# CVaR Calculation
eCVaR=np.zeros((n_ret_select))
for i in range(n_ret_select):
    select=ret_select[:,i]<np.quantile(ret_select[:,i],0.05)
    eCVaR[i]=np.mean(ret_select[select,i])
eCVaR=abs(eCVaR)

# Equal risk CVaR weights calculation
w_eCVaR_port=(1/eCVaR)/(np.sum(1/eCVaR))
hd_eCVaR = 1/(np.sum(w_eCVaR_port**2))
ret_eCVaR_port=np.dot(ret_select,w_eCVaR_port)


### Long/short "alpha" equity portfolio
## Long/Short MV weights
# Long
w_L=np.zeros((n_assets))
cov_L=np.cov(ret[:,alpha_best],rowvar=0)
w_L[alpha_best]=MV(cov_L)
# Short
w_S=np.zeros((n_assets))
cov_S=np.cov(ret[:,alpha_worst],rowvar=0)
w_S[alpha_worst]=MV(cov_S)

## Beta neutral
#Long
ret_L=np.dot(ret,w_L.T)
cov_L_port=np.cov((ret_L,ret_bm))
beta_L=cov_L_port[0,1]/cov_L_port[1,1]
#Short
ret_S=np.dot(ret,w_S.T)
cov_S_port=np.cov((ret_S,ret_bm))
beta_S=cov_S_port[0,1]/cov_S_port[1,1]

## Hedge ratio for Long/Short Portfolio
hedge_ratio=beta_L/beta_S
# Allocations to Long/Short portfolios
allocation_L=1/(1+hedge_ratio)
allocation_S=hedge_ratio*allocation_L
# Beta Long/Short allocations
print(beta_L*allocation_L)
print(beta_S*allocation_S)

## Returns Long/Short Portfolio
w_L_port=(w_L*allocation_L)
w_S_port=(w_S*allocation_S)
w_LS_port=np.sum((w_L_port,w_S_port),axis=0)
ret_ls_port = np.sum(np.prod((ret,w_LS_port.T)),axis=1)


### Stress testing
## Simulation of shock diffusion
nsim=190
shock=-0.1
ret_stress = np.column_stack((ret_bm, ret_erc_port, ret_eCVaR_port, ret_ls_port))
std_stress=np.std(ret_stress,axis=0)
w_stress = (1/std_stress)/np.sum(1/std_stress)
n_assets_stress = ret_stress.shape[1]
sim_shocks=np.zeros((nsim, n_assets_stress))
annualized_loss_shock = np.zeros((1,n_assets_stress))
sim_shocks[:,0]=shock
for i in range(nsim):
    sim=resample(ret_stress,replace=True)
    for j in range(n_assets_stress):
        if j!=0:
            model = sm.OLS(sim[:,j],sm.add_constant(sim[:,0]))
            results = model.fit()
            betas=results.params 
            sim_shocks[i,j]=betas[0]+betas[1]*shock

## Expected losses
loss_shock_mean = np.mean(sim_shocks, axis=0)
for i in range(n_assets_stress):
    annualized_loss_shock[:,i] = (1 + np.mean(sim_shocks[:,i]))**12 - 1

port_shocks=np.prod((sim_shocks,w_stress.T))
port_index=np.cumprod((1+port_shocks),axis=1)
sim_shocks_cum_ret = np.cumprod((1+sim_shocks),axis=1)
sim_shocks_cum_loss = np.cumprod((1+sim_shocks),axis=1)-1

unique_port_shocks = np.dot(sim_shocks,w_stress.T)

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle('Portfolio Simulation Shocks Distribution')

sns.distplot(unique_port_shocks, hist=True, kde=True, bins=50, ax=axes[0,0])
axes[0,0].set_title('Portfolio of Portfolios Distribution')
# Distribution "alpha ERC" equity portfolio
sns.distplot(port_shocks[:,1], hist=True, kde=True, bins=50, ax=axes[0,1])
axes[0,1].set_title('“Alpha ERC” Portfolio')
# Distribution "alpha eCVaR" equity portfolio
sns.distplot(port_shocks[:,2], hist=True, kde=True, bins=50, ax=axes[1,0])
axes[1,0].set_title('“Alpha eCVaR” Portfolio')
# Distribution Long/Short "alpha" equity portfolio
sns.distplot(port_shocks[:,3], hist=True, kde=True, bins=50, ax=axes[1,1])
axes[1,1].set_title('“Alpha Long/Short" Portfolio')

# CI 95%
ci_95= np.percentile(sim_shocks, (2.5,97.5), axis=0)


# Cumulative returns and losses for alpha ERC portfolio
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.plot(sim_shocks_cum_ret[:,1])
ax1.set_xlabel('Date')
ax1.set_ylabel("Cumulative Returns")
ax1.set_title("alpha ERC Cumulative Returns")
plt.show();

fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.plot(sim_shocks_cum_loss[:,1])
ax1.set_xlabel('Date')
ax1.set_ylabel("Cumulative Returns")
ax1.set_title("alpha ERC Cumulative Losses")
plt.show();

# Cumulative returns and losses for alpha eCVaR portfolio
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.plot(sim_shocks_cum_ret[:,2])
ax1.set_xlabel('Date')
ax1.set_ylabel("Cumulative Returns")
ax1.set_title("alpha eCVaR Cumulative Returns")
plt.show();

fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.plot(sim_shocks_cum_loss[:,2])
ax1.set_xlabel('Date')
ax1.set_ylabel("Cumulative Returns")
ax1.set_title("alpha eCVaR Cumulative Losses")
plt.show();

# Cumulative returns and losses for alpha Long/Short portfolio
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.plot(sim_shocks_cum_ret[:,3])
ax1.set_xlabel('Date')
ax1.set_ylabel("Cumulative Returns")
ax1.set_title("alpha Long/Short Cumulative Returns")
plt.show();

fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.plot(sim_shocks_cum_loss[:,3])
ax1.set_xlabel('Date')
ax1.set_ylabel("Cumulative Returns")
ax1.set_title("alpha Long/Short Cumulative Losses")
plt.show();















