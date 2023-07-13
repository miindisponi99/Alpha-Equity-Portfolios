# Alpha-Equity-Portfolios

## Weighting “Alpha” equity portfolios
### “Alpha ERC” equity portfolio
Equal Risk Contribution strategies aim at building balanced portfolios in terms of contribution to risk by the various assets and these lead to portfolios in which less weight is assigned to risky assets. Furthermore, the risk parity strategy does not aim so much to increase the performance in absolute terms (the performance is generally not particularly high) as to improve the behavior of the portfolio in the negative phases of the markets.<br>
To build the “alpha ERC” equity portfolio, we used the stocks with the best 15 alphas out of the 48 available in the Eurostoxx50 (equivalent to the ones that are above 0.6875 quantile); we determined them by using the following formula and by making the assumption that the error term 𝜀! (which represents the deviation from the best-fitting line) is 0 on average:
<div style="text-align: center;">
  α_s = R_{s,t} - β_s R_{STOXX50E,t}<br>
where R",! and R$%&''()*,! have been calculated taking into account annualized returns,
whereas 𝛽" calculation was straightforward as we computed the covariance between the stock’s returns and the benchmark, and we divided it by the variance of the benchmark
</div><br>
Then, we computed the covariance matrix and, by applying the ERC formula (in which positivity and unitary budget constraints are showed) we got the weights for our “alpha ERC” equity portfolio. Moreover, we computed the returns of the portfolio by multiplying these weights with the best 15 alpha stocks’ returns. Regarding the obtained results, what can be said is that: it emerges that the greater the volatility of an asset the lower the weight assigned to that specific asset and the greater the correlation the lower the weight given. In this “alpha ERC” equity portfolio, the stock with the highest weight is Koninklijke Ahold Delhaize N (11.09%), whilst the one with the lowest weight is Volkswagen Ag-pref (4.20%). Furthermore, we wanted to calculate the inverse of the Herfindahl index so as to have the number of effective stocks in which we would invest using this portfolio and the result is 13.81 stocks out of 48.<br><br>

### “Alpha CVaR” equity portfolio
In order to have the “alpha CVaR” equity portfolio, we took into consideration the aforementioned formula to find individual stocks’ alpha. Therefore, we first selected the 15 highest alpha stocks and then we used the formula for the Conditional Value-at-Risk by selecting the stocks crossing the threshold at 95% confidence level. From this, we computed the mean of those stocks returns and put the value of the equal CVaR in absolute terms as to display the magnitude of the eCVaR. This allowed us to find the weights that contribute in equal terms to the CVaR which was found by dividing the reciprocal of the eCVaR of each stock by the sum of the stocks’ reciprocal eCVaR. Lastly, the returns of the “alpha CVaR” equity portfolio were computed by multiplying the returns by these newly found weights. Here, we found that the stock with the highest weight is represented by Unilever NV (9.89%) and the one which has the lowest weight is still Volkswagen Ag-pref (3.51%). As for the “alpha ERC” equity portfolio, we wanted to compute the inverse of the Herfindahl index and we found that we would invest in 13.98 effective stocks out of the 48 available; this result is similar to the one shown in the “alpha ERC” equity portfolio.<br><br>

### Long/Short “alpha” equity portfolio
To find alphas for the individual stocks of the long/short “alpha” equity portfolio, we made the same calculation as for the other two aforementioned portfolios. However, here we included also the worst 15 alphas as we needed the “short” allocation in the portfolio. Since


Applied Equal Risk Contribution, Conditional Value-at-Risk and Long/Short strategies to “alpha” equity portfolios and undertook a theoretical stress test by performing a Monte Carlo simulation with bootstrapping
