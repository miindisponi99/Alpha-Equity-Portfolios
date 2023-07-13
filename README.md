# Analysis of Alpha Equity Portfolios

## Weighting “Alpha” equity portfolios
### “Alpha ERC” equity portfolio
Equal Risk Contribution strategies aim at building balanced portfolios in terms of contribution to risk by the various assets and these lead to portfolios in which less weight is assigned to risky assets. Furthermore, the risk parity strategy does not aim so much to increase the performance in absolute terms (the performance is generally not particularly high) as to improve the behavior of the portfolio in the negative phases of the markets.<br>
To build the “alpha ERC” equity portfolio, we used the stocks with the best 15 alphas out of the 48 available in the Eurostoxx50 (equivalent to the ones that are above 0.6875 quantile); we determined them by using the following formula and by making the assumption that the error term $𝜀_t$ (which represents the deviation from the best-fitting line) is 0 on average:
<div style="text-align: center;">
  __$α_s  = R_(s,t)  - β_s R_(STOXX50E,t)$<br>__
where $R_(s,t)$ and $R_STOXX50E,t$ have been calculated taking into account annualized returns,
whereas $β_s$ calculation was straightforward as we computed the covariance between the stock’s returns and the benchmark, and we divided it by the variance of the benchmark
</div><br><br>
Then, we computed the covariance matrix and, by applying the ERC formula (in which positivity and unitary budget constraints are showed) we got the weights for our “alpha ERC” equity portfolio. Moreover, we computed the returns of the portfolio by multiplying these weights with the best 15 alpha stocks’ returns. Regarding the obtained results, what can be said is that: it emerges that the greater the volatility of an asset the lower the weight assigned to that specific asset and the greater the correlation the lower the weight given. In this “alpha ERC” equity portfolio, the stock with the highest weight is Koninklijke Ahold Delhaize N (11.09%), whilst the one with the lowest weight is Volkswagen Ag-pref (4.20%). Furthermore, we wanted to calculate the inverse of the Herfindahl index so as to have the number of effective stocks in which we would invest using this portfolio and the result is 13.81 stocks out of 48.<br><br>

### “Alpha CVaR” equity portfolio
In order to have the “alpha CVaR” equity portfolio, we took into consideration the aforementioned formula to find individual stocks’ alpha. Therefore, we first selected the 15 highest alpha stocks and then we used the formula for the Conditional Value-at-Risk by selecting the stocks crossing the threshold at 95% confidence level. From this, we computed the mean of those stocks returns and put the value of the equal CVaR in absolute terms as to display the magnitude of the eCVaR. This allowed us to find the weights that contribute in equal terms to the CVaR which was found by dividing the reciprocal of the eCVaR of each stock by the sum of the stocks’ reciprocal eCVaR. Lastly, the returns of the “alpha CVaR” equity portfolio were computed by multiplying the returns by these newly found weights. Here, we found that the stock with the highest weight is represented by Unilever NV (9.89%) and the one which has the lowest weight is still Volkswagen Ag-pref (3.51%). As for the “alpha ERC” equity portfolio, we wanted to compute the inverse of the Herfindahl index and we found that we would invest in 13.98 effective stocks out of the 48 available; this result is similar to the one shown in the “alpha ERC” equity portfolio.<br><br>

### Long/Short “alpha” equity portfolio
To find alphas for the individual stocks of the long/short “alpha” equity portfolio, we made the same calculation as for the other two aforementioned portfolios. However, here we included also the worst 15 alphas as we needed the “short” allocation in the portfolio. Since we wanted to perform the Minimum Volatility portfolio, we computed the covariance matrix for the best and worst 15 alphas and then we utilized an optimizer to minimize the covariance among stocks and find weights for the long portfolio as well as for the short one; these two MV portfolios have been subject to positivity and unitary budget constraints.
Given the fact that the Long/Short portfolio is determined by:<br>
__Long/Short portfolio = a(longportfolio) - b(shortportfolio)__<br>
with a and b determined so that beta of the long and short portfolios to the Eurostoxx50 are equal<br><br>
Indeed, to have the “beta neutral” constraint we first computed the covariance of the long and short portfolios’ returns (calculated with the new determined weights) and benchmark returns. Then, for long and short portfolios we calculated beta which are 0.52 and 0.59, respectively. We found the hedge ratio by dividing long portfolio beta and short portfolio beta and this is equal to 0.89 and it is useful to determine the optimal allocation for the long and short portfolios given a leverage of 1 (because we wanted a and b greater than 0 and a+b = 2). It resulted in a long allocation of 105.73%, and 94.27% as a short allocation. Eventually, we adjusted the portfolio’s weights by multiplying them with the allocation based on the leverage, and we also found Long/Short “alpha” equity portfolio returns.<br>
Here, the stock with the highest weight is Danone (20.54%) and the one with the lowest one is ASML Holding NV (5.01522e-19).<br>

### Why the “alpha” portfolios’ weights differ?
ERC portfolio weights are more based on covariance among stocks and are determined so that their risk contribution to the portfolio is equal among assets chosen, i.e. best alpha stocks. As regards the equal Conditional Value-at-Risk, weights differ from the former portfolio as we calculated them based on the expected shortfall of returns at 95% confidence level; anyway, these weights are similar to the ERC portfolio weights because taking 95% confidence level we aim at “equaling” risk of the individual stocks. Finally, for the Long/Short portfolio, weights are much different from the two aforementioned portfolios as we include the stocks with the lowest 15 alphas. If we look at the Long portfolio allocation, we can state that their allocation is much based on the hedge ratio (between beta of the long and short portfolios) and this causes weights to change dramatically in the final Long/Short portfolio.<br><br>

## Stress-tests of “Alpha” equity portfolios
Assuming a monthly drop of -10% in returns of the benchmark Eurostoxx50 index, we performed stress-tests for our “alpha” equity portfolios by making a Monte Carlo simulation with bootstrapping, so by resampling returns and simulating expected losses for each portfolio (with 190 simulations). Then, we also wanted to see distribution of expected losses for each “alpha” equity portfolio and for a unique portfolio of portfolios where weights are calculated dividing the reciprocal of the standard deviation of each portfolio by the sum of the reciprocal of the portfolios’ standard deviation.<br>
Hereafter, we have the analysis of the expected losses and confidence intervals of the former for each portfolio under stress-testing:
1. _“alpha ERC” equity portfolio_: the average expected losses under stress-test is -6.55%, and the annualized loss is -55.64%. Moreover, if we look at 95% confidence interval, making the assumption of potential estimation errors on the estimation of the estimated $𝛽_s$ parameters, we have: high CI (at 97.5% percentile) is -5.77%, and low CI (at 2.5% percentile) is -7.35%. We wanted also the cumulative returns and losses therefore we created some graphs showing how this stress-test affected our portfolio returns
2. _“alpha” eCVaR equity portfolio_: here we found close results to the “alpha” ERC portfolio as regards average expected losses and annualized loss, that in this portfolio are -6.63% and -56.08%, respectively. Even analyzing 95% confidence interval, where we have high CI at -5.78% and low CI at -7.45%, we arrive to similar estimations
3. _“alpha” Long/Short equity portfolio_: computing the expected losses for this portfolio, the average loss would be -5.01% and the annualized one would be -46.02%, but at 95% CI, the 97.5% percentile is -4.16% and the 2.5% percentile is -5.77%
<br>

### Comments about stress-tests for “alpha” equity portfolios
Thanks to stress testing, we cannot still state which portfolio is the best to follow, but during a monthly drop of -10% the “alpha ERC” and “alpha eCVaR” portfolios had lower cumulative losses (and therefore higher cumulative returns) than “alpha” Long/Short portfolio. Nevertheless, the latter showed better 95% confidence intervals and lower average losses.
