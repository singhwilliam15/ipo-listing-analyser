# What separated IPOs that listed up from those that listed down

392 mainboard IPOs listed 2022 to 2026 with a listing-day price (NSE data; fundamentals read from each RHP).
"Gain" means the listing-day open was above the issue price: 285 IPOs (average +26.6%).
"Loss" means flat or lower: 107 IPOs (average -5.6%).

| Feature | Gainers (median) | Losers (median) | Signal |
|---|---|---|---|
| QIB subscription | 90x | 4.2x | strong (Spearman 0.74) |
| Total subscription | 55x | 3.0x | strong (0.73) |
| Retail subscription | 13.3x | 2.3x | strong (0.59) |
| PAT margin | 8.9% | 6.0% | modest |
| Revenue CAGR | 27.9% | 21.3% | modest |
| Return on net worth | 22% | 17% | modest |
| Debt / equity | 0.47 | 0.80 | weak |
| P/E premium to peers | -12.7% | +1.4% | weak |
| Issue size, OFS share, promoter holding, anchor share, P/E | no difference | | none |

## Gainers
- QIB subscription of 50x or more: 95% gained (179 IPOs). Total subscription over 100x: all 62 gained, +49% on average.
- Debt/equity of 1 or less: 79% gained. P/E 20% or more below peers: 80% (102 IPOs).
- IPO priced above the pre-IPO placement: 84% gained (55 IPOs).
- Strong sectors: internet and platforms (86%), FMCG (83%), healthcare and financial services (82%), industrials (80%), consumer (79%).

## Losers
- 53 of 107 losers were under 3x subscribed; 73 were under 10x. With QIB subscription under 5x only 42% gained.
- Debt/equity above 2: 51% gained. Loss-making companies: median listing gain +2.0% vs +10.5%.
- Weak sectors: media and education (44%), banking and NBFC (50%), logistics and travel (50%), auto (56%).
- Exceptions exist: Shree Ram Twistex (44x), Udayshivakumar Infra (31x) and R K Swamy (26x) were well subscribed and still lost.

## By year
| Year | IPOs | Listed above issue | Median gain |
|---|---|---|---|
| 2022 | 38 | 66% | +3.4% |
| 2023 | 57 | 81% | +17.1% |
| 2024 | 87 | 78% | +16.8% |
| 2025 | 103 | 70% | +6.0% |
| 2026 | 107 | 69% | +7.6% |

## Where this data disagrees with commentary
Analysts often blame OFS-heavy and aggressively priced issues, and small issues. In this sample OFS share and issue
size made no clear difference, and a high P/E alone did not predict a loss.

## Backtest

The dataset has 394 mainboard IPOs (392 with a listing price). Five that NSE's past-issues list omits (Ola Electric, IREDA,
NTPC Green, Tata Capital, Aegis Vopak) were added from their NSE issue pages, DCX Systems and NSDL got their listing dates,
and IREDA's subscription was entered by hand from published figures.

**Which models the numbers describe.** Live estimates in the app use models refit on every IPO to date, which cannot be tested
out of sample. Their accuracy is measured by the walk-forward below, where each test year is scored by models trained only on
earlier years. A separate single test (trained 2022-2025, scored on 2026) is also shown.

### Walk-forward: every year scored by models trained only on earlier years

A single test year can flatter a model, so each of 2024, 2025 and 2026 is scored by models fitted on all earlier years
(`R.walk_forward`). Pooled over three years the median gain error of Model A is 9.4 points; 2026 alone (6.6) was the easiest year.

**Chance of a listing gain once subscription is final** (log QIB, total, retail subscription and PAT margin)

| Test year | IPOs | AUC | Brier | Accuracy | Base rate |
|---|---|---|---|---|---|
| 2024 (trained 2022-23) | 87 | 0.85 | 0.135 | 81% | 78% |
| 2025 (trained 2022-24) | 103 | 0.81 | 0.168 | 73% | 70% |
| 2026 (trained 2022-25) | 107 | 0.85 | 0.147 | 81% | 69% |
| **Pooled 2024-2026** | 297 | 0.83 | 0.151 | 78% | 72% |

Pooled it reaches AUC 0.83 and 78% accuracy against a 72% base rate.

**Before bidding opens: the new pre-bid method against the old one**

The old method was a logistic regression on fundamentals, with the 10 nearest past IPOs as the gain range. The new method
forecasts the IPO's final QIB subscription from the same fundamentals (a ridge regression fitted inside the training rows; its
residual SD comes from 5-fold out-of-fold predictions), spreads the forecast into 21 points (mean +/- 2.5 residual SD, weighted
by the normal density), pushes each point through the demand model and Model A (total and retail subscription from their usual
relationship with QIB), and weights the answers. The gain range is the weighted mixture, widened by walk-forward residuals.
The 10 nearest IPOs stay in the app as a table only.

| Test year | New AUC | New Brier | New accuracy | Old AUC | Old Brier | Old accuracy |
|---|---|---|---|---|---|---|
| 2024 (trained 2022-23) | 0.63 | 0.171 | 76% | 0.61 | 0.173 | 79% |
| 2025 (trained 2022-24) | 0.59 | 0.207 | 69% | 0.59 | 0.209 | 71% |
| 2026 (trained 2022-25) | 0.67 | 0.202 | 71% | 0.68 | 0.200 | 70% |
| **Pooled 2024-2026** | 0.63 | 0.195 | 72% | 0.61 | 0.195 | 73% |

On probability the two methods are the same: pooled AUC 0.63 against 0.61, and the new
one is slightly less accurate at a 0.5 cut-off (72% against 73%). The gain range is where it helps: pooled median error 10.6 points
against 13.6, rank correlation 0.28 against 0.17, with the widened ranges covering 51% and 79%
(targets 50% and 80%).

How good is the QIB forecast itself? Poor, which is the reason pre-bid ranges stay wide:

| Test year | IPOs | Error (RMSE) | R² vs average | Correlation |
|---|---|---|---|---|
| 2024 (trained 2022-23) | 84 | 1.33 | 0.14 | 0.52 |
| 2025 (trained 2022-24) | 103 | 1.58 | 0.03 | 0.25 |
| 2026 (trained 2022-25) | 106 | 1.71 | 0.09 | 0.30 |
| **Pooled 2024-2026** | 293 | 1.56 | 0.10 | 0.34 |

Fundamentals explain about a tenth of the variation in how much QIB demand an IPO draws. The assumed spread
(residual SD 1.48) is a little tight for 2025-2026, where the realised error was larger; the range
widening above absorbs that.

**Why pre-bid probabilities below 80% are shown as "about the base rate".** Pooled out-of-sample, only 54 of 297 pre-bid
probabilities reached 80%, and 89% of those listed above issue. The other 243 listed above issue 68% of the time against a
72% base rate, so below 80% the model adds nothing a stated base rate does not. The app shows the base rate from the training rows.

**Expected listing gain (%)**

| Model | Test year | IPOs | Median error (pts) | Spearman | 25-75 raw | 25-75 widened | 10-90 raw | 10-90 widened |
|---|---|---|---|---|---|---|---|---|
| A: quantile regression (final figures) | 2024 | 84 | 12.8 | 0.73 | 45% | 48% | 76% | 77% |
| A: quantile regression (final figures) | 2025 | 103 | 10.1 | 0.71 | 48% | 49% | 70% | 76% |
| A: quantile regression (final figures) | 2026 | 106 | 6.6 | 0.77 | 56% | 57% | 76% | 89% |
| A: quantile regression (final figures) | **pooled** | 293 | 9.4 | 0.73 | 50% | 51% | 74% | 81% |
| B: similar QIB (final figures) | 2024 | 84 | 9.9 | 0.77 | 48% | 49% | 79% | 80% |
| B: similar QIB (final figures) | 2025 | 103 | 10.8 | 0.64 | 49% | 49% | 74% | 74% |
| B: similar QIB (final figures) | 2026 | 106 | 7.1 | 0.75 | 53% | 61% | 82% | 88% |
| B: similar QIB (final figures) | **pooled** | 293 | 9.1 | 0.70 | 50% | 53% | 78% | 81% |
| Pre-bid, new: QIB forecast | 2024 | 87 | 16.1 | 0.40 | 41% | 43% | 76% | 76% |
| Pre-bid, new: QIB forecast | 2025 | 103 | 9.0 | 0.08 | 51% | 61% | 81% | 85% |
| Pre-bid, new: QIB forecast | 2026 | 107 | 10.0 | 0.29 | 44% | 49% | 73% | 75% |
| Pre-bid, new: QIB forecast | **pooled** | 297 | 10.6 | 0.28 | 46% | 51% | 76% | 79% |
| Pre-bid, old: 10 nearest IPOs | 2024 | 79 | 17.7 | 0.10 | 46% | 52% | 68% | 81% |
| Pre-bid, old: 10 nearest IPOs | 2025 | 103 | 13.6 | 0.23 | 41% | 54% | 66% | 84% |
| Pre-bid, old: 10 nearest IPOs | 2026 | 105 | 12.2 | 0.24 | 35% | 44% | 64% | 76% |
| Pre-bid, old: 10 nearest IPOs | **pooled** | 287 | 13.6 | 0.17 | 40% | 50% | 66% | 80% |

"Raw" is the model's own p25-p75 and p10-p90 range. "Widened" adds the amount in the table below to each side of the range and is
scored leave-one-year-out: the widening for a year comes only from the other test years' residuals, so it is never scored on its own
data. Targets are 50% and 80%.

**Stored widening, by QIB band** (fitted on 2024-2025 residuals for the published models; production models also use 2026)

| Model | QIB band | IPOs behind it | Widen 25-75 (pts each side) | Widen 10-90 (pts each side) |
|---|---|---|---|---|
| A: quantile regression (final figures) | <10x | 50 | +0.2 | +2.6 |
| A: quantile regression (final figures) | 10-100x | 65 | +0.3 | +0.2 |
| A: quantile regression (final figures) | >100x | 72 | +2.8 | +6.7 |
| B: similar QIB (final figures) | <10x | 50 | +1.1 | +2.4 |
| B: similar QIB (final figures) | 10-100x | 65 | +0.0 | +0.0 |
| B: similar QIB (final figures) | >100x | 72 | +7.8 | +5.7 |
| Pre-bid, new: QIB forecast | pre-bid | 190 | +0.8 | +1.6 |
| Pre-bid, old: 10 nearest IPOs | pre-bid | 182 | +2.2 | +5.2 |

**Isotonic calibration of the subscription model, before and after.** Each year's probabilities were mapped by an isotonic
regression fitted on the other years' held-out predictions only.

| Test year | Brier before | Brier after | Calibration error before | after | Accuracy before | after |
|---|---|---|---|---|---|---|
| 2024 (trained 2022-23) | 0.135 | 0.137 | 0.094 | 0.103 | 81% | 80% |
| 2025 (trained 2022-24) | 0.168 | 0.169 | 0.125 | 0.086 | 73% | 72% |
| 2026 (trained 2022-25) | 0.147 | 0.163 | 0.097 | 0.121 | 81% | 77% |
| **Pooled 2024-2026** | 0.151 | 0.158 | 0.066 | 0.066 | 78% | 76% |

Pooled calibration by decile (30 IPOs each):

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Predicted, before | 34% | 44% | 53% | 71% | 83% | 88% | 92% | 94% | 95% | 96% |
| Actual, before | 40% | 31% | 52% | 69% | 63% | 76% | 93% | 97% | 100% | 100% |
| Predicted, after | 32% | 39% | 47% | 63% | 75% | 78% | 94% | 99% | 100% | 100% |
| Actual, after | 40% | 34% | 55% | 69% | 57% | 79% | 100% | 90% | 100% | 97% |

It does not help out of sample. Pooled Brier goes from 0.151 to 0.158 and accuracy from 78% to 76%, and
calibration error is unchanged. With a few hundred IPOs the isotonic step function overfits, so it moves the middle deciles as
often in the wrong direction as the right one. The code therefore applies it only if it lowers the walk-forward Brier score by at
least 0.005; it does not, for the published models (0.153 to 0.152) or the production models, so the app shows the
subscription model's probabilities as fitted. Set `ISOTONIC = "on"` in `src/report.py` to force it.

### Single test: trained 2022-2025, scored on 2026 (107 IPOs)

| Model | IPOs | AUC | Brier | Accuracy at 0.5 | Base rate |
|---|---|---|---|---|---|
| Subscription model, final figures | 106 | **0.85** | 0.147 | **81%** | 69% |
| Pre-bid, new (QIB forecast) | 107 | 0.67 | 0.202 | 71% | 69% |
| Pre-bid, old (logistic on fundamentals) | 107 | 0.68 | 0.200 | 70% | 69% |

Baselines on the same IPOs: always apply 69%, total subscription over 10x 73%.

Calibration of the subscription model on 2026 (deciles, about 10 IPOs each):

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Predicted | 32% | 39% | 45% | 54% | 71% | 84% | 90% | 93% | 94% | 95% |
| Actual | 27% | 45% | 10% | 73% | 80% | 82% | 80% | 91% | 100% | 100% |

The QIB forecast on 2026: R² 0.11 against the training average, error 1.71 (assumed spread 1.48).

### While bidding is open

Live subscription is a poor guide until the books close, because QIBs bid mostly on the last day. The app keeps the pre-bid
estimate as the headline during bidding, shows the subscription model only as "if bidding closed at today's figures", leads with
the QIB subscription needed for a 70/85/95% chance, and hides allotment odds and profit per lot until bidding has closed.
`src/qibforecast.py` defines a pluggable forecaster; a stub documents how to condition it on the Day-1 / Day-2 QIB that
`python -m src.build snapshot` is now logging, once there is enough history.

### Strategies on 2026

| Rule | IPOs applied | Hit rate | Avg listing gain | Expected gain per application | Summed expected gain per lot |
|---|---|---|---|---|---|
| Always apply | 107 | 69% | +14.4% | +0.5% | ₹7,624 |
| Total subscription > 10x | 59 | 88% | +26.5% | +1.2% | ₹10,473 |
| QIB > 3x | 82 | 80% | +19.6% | +1.2% | ₹14,403 |
| QIB > 5x | 68 | 88% | +24.2% | +1.5% | ₹15,346 |
| QIB > 10x | 62 | 89% | +25.9% | +1.3% | ₹11,854 |
| QIB > 20x | 55 | 91% | +28.8% | +1.5% | ₹12,149 |
| QIB > 50x | 43 | 91% | +33.6% | +1.4% | ₹8,960 |

Expected gain per application multiplies each IPO's gain by the chance of allotment, min(1, 1 / retail subscription), so
heavily subscribed IPOs, where the gain is large but allotment is rare, count for less. A higher hit rate does not mean more
money per application: summed over the year, **QIB > 5x** earned the most per lot.

### Caveats
- A test year holds 87-107 IPOs, so each decile and each year's coverage rests on about ten to a hundred IPOs. The pooled
  walk-forward is the fairer summary.
- The subscription model needs the final figures. Before the close it is a floor.
- The QIB forecast is weak (R² about 0.1). The pre-bid gain range is better centred than before (median error 10.6 against
  13.6 points) but is still wide, and the pre-bid probability is close to the base rate.
- Chittorgarh's mainboard list could not be fetched (its tables load through JavaScript), so the IPO list was checked only
  against NSE. Yearly mainboard counts in the data: 2022: 38, 2023: 57, 2024: 88, 2025: 104, 2026: 107 so far.

## Outside commentary
- Subscription (especially QIB) and grey-market premium are the factors most often tied to listing returns; see
  [Angel One](https://www.angelone.in/knowledge-center/ipo/ipo-oversubscription-versus-listing-gains) and
  [one blog analysis of 300 Indian IPOs](https://dev-ritik.github.io/blog/2025/12/31/i-analyzed-300-indian-ipos-heres-what-actually-works).
- 2025 weakness is attributed to aggressive pricing, heavy OFS, loss-making new-age issuers and supply:
  [Business Standard](https://www.business-standard.com/amp/markets/ipo/three-of-four-new-listings-flop-on-debut-in-2025-125093000900_1.html),
  [Business Standard](https://www.business-standard.com/amp/markets/news/ipos-in-h1-2025-were-priced-to-perfection-be-selective-now-say-analysts-125062600755_1.html).
- Listing gains do not always last:
  [Business Today](https://www.businesstoday.in/personal-finance/story/ipos-delivered-21-average-listing-gains-but-did-investors-actually-make-money-565507-2026-11-04).
