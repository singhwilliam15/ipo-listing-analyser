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

### Walk-forward: every year scored by models trained only on earlier years

A single test year can flatter a model, so each of 2024, 2025 and 2026 is scored by models fitted on all earlier years
(`R.walk_forward`). Nothing in a year's score was seen when its models were fitted. The earlier single-year claim of a
6.6-point median gain error was 2026 alone, which turned out to be the easiest year; pooled over three years the figure is
9.4 points.

**Chance of a listing gain, P(open > issue price)**

| Test year (trained on all earlier years) | IPOs | Subscription model AUC | Brier | Accuracy | Fundamentals-only AUC | Accuracy | Base rate |
|---|---|---|---|---|---|---|---|
| 2024 (trained 2022-23) | 87 | 0.85 | 0.135 | 81% | 0.61 | 79% | 78% |
| 2025 (trained 2022-24) | 103 | 0.81 | 0.168 | 73% | 0.59 | 71% | 70% |
| 2026 (trained 2022-25) | 107 | 0.85 | 0.147 | 81% | 0.68 | 70% | 69% |
| **Pooled 2024-2026** | 297 | 0.83 | 0.151 | 78% | 0.61 | 73% | 72% |

The subscription model uses log QIB, total and retail subscription plus PAT margin, scored on the final figures. Pooled it
reaches AUC 0.83 and 78% accuracy against a 72% base rate. Fundamentals alone (what exists before
bidding opens) reach only AUC 0.61: close to a coin flip, and it does not improve over time.

**Expected listing gain (%)**

| Model | Test year | IPOs | Median error (pts) | Spearman | 25-75 covers, raw | 25-75, widened | 10-90 covers, raw | 10-90, widened |
|---|---|---|---|---|---|---|---|---|
| A: quantile regression | 2024 | 84 | 12.8 | 0.73 | 45% | 48% | 76% | 77% |
| A: quantile regression | 2025 | 103 | 10.1 | 0.71 | 48% | 49% | 70% | 76% |
| A: quantile regression | 2026 | 106 | 6.6 | 0.77 | 56% | 57% | 76% | 89% |
| A: quantile regression | **pooled** | 293 | 9.4 | 0.73 | 50% | 51% | 74% | 81% |
| B: similar QIB | 2024 | 84 | 9.9 | 0.77 | 48% | 49% | 79% | 80% |
| B: similar QIB | 2025 | 103 | 10.8 | 0.64 | 49% | 49% | 74% | 74% |
| B: similar QIB | 2026 | 106 | 7.1 | 0.75 | 53% | 61% | 82% | 88% |
| B: similar QIB | **pooled** | 293 | 9.1 | 0.70 | 50% | 53% | 78% | 81% |
| Pre-bid: 10 nearest IPOs | 2024 | 79 | 17.7 | 0.10 | 46% | 52% | 68% | 81% |
| Pre-bid: 10 nearest IPOs | 2025 | 103 | 13.6 | 0.23 | 41% | 54% | 66% | 84% |
| Pre-bid: 10 nearest IPOs | 2026 | 105 | 12.2 | 0.24 | 35% | 44% | 64% | 76% |
| Pre-bid: 10 nearest IPOs | **pooled** | 287 | 13.6 | 0.17 | 40% | 50% | 66% | 80% |

"Raw" is the model's own p25-p75 and p10-p90 range. "Widened" adds the amount in the table below to each side of the range
and is scored by leave-one-year-out: the widening for a year comes only from the other test years' residuals, so it is never
scored on its own data. Targets are 50% and 80%. Pooled, Model A's widened ranges cover 51% and 81%
(raw 50% and 74%), and the pre-bid range 50% and 80%
(raw 40% and 66%).

**Stored widening, by QIB band** (fitted on 2024-2025 residuals for the published models; the production models also use 2026)

| Model | QIB band | IPOs behind it | Widen 25-75 by (pts each side) | Widen 10-90 by (pts each side) |
|---|---|---|---|---|
| A: quantile regression | <10x | 50 | +0.2 | +2.6 |
| A: quantile regression | 10-100x | 65 | +0.3 | +0.2 |
| A: quantile regression | >100x | 72 | +2.8 | +6.7 |
| B: similar QIB | <10x | 50 | +1.1 | +2.4 |
| B: similar QIB | 10-100x | 65 | +0.0 | +0.0 |
| B: similar QIB | >100x | 72 | +7.8 | +5.7 |
| Pre-bid: 10 nearest IPOs | pre-bid | 182 | +2.2 | +5.2 |

Model A and Model B trade places by year; neither is clearly better, which is why the default is chosen on the last full year
inside each model's own training data rather than on its test year.

### Published test: trained 2022-2025, scored on 2026 (107 IPOs)

These are the scores the app shows for the published models. Live estimates use models refit on every row including 2026, so
their own accuracy cannot be measured; the published and walk-forward figures are the evidence for them.

| Model | IPOs | AUC | Brier | Accuracy at 0.5 | Base rate |
|---|---|---|---|---|---|
| Subscription model, final figures | 106 | **0.85** | 0.147 | **81%** | 69% |
| Fundamentals only (pre-bid) | 107 | 0.68 | 0.200 | 70% | 69% |

Baselines on the same IPOs: always apply 69%, total subscription over 10x 73%.

Calibration of the subscription model on 2026 (deciles of predicted probability, about 10 IPOs each):

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Predicted | 32% | 39% | 45% | 54% | 71% | 84% | 90% | 93% | 94% | 95% |
| Actual | 27% | 45% | 10% | 73% | 80% | 82% | 80% | 91% | 100% | 100% |

It ranks well and the top deciles are close to calibrated. The middle is noisy because each decile holds only about 10 IPOs.
The fundamentals-only model is overconfident at the low end: its lowest decile predicted 57% and 45% actually gained.

### While bidding is open

Live subscription is a poor guide until the books close, because QIBs bid mostly on the last day. The app therefore keeps the
pre-bid estimate as the headline during bidding, shows the subscription model only as "if bidding closed at today's figures",
leads with the QIB subscription needed for a 70/85/95% chance, and hides allotment odds and profit per lot until bidding has
closed.

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
money per application: summed over the year, **QIB > 5x** earned the most per lot, more than the strictest rule, because the strongest
IPOs are the hardest to get allotted.

### Caveats
- A test year holds 87-107 IPOs, so each decile and each year's coverage rests on about ten to a hundred IPOs. The pooled
  walk-forward is the fairer summary.
- The subscription model needs the final figures. Before the close it is a floor.
- Published scores come from models trained to 2025. The models behind live estimates are refit on every row, so nothing
  tests them out of sample.
- Chittorgarh's mainboard list could not be fetched (its tables load through JavaScript), so the IPO list was checked only
  against NSE. Yearly mainboard counts in the data: 2022: 38, 2023: 57, 2024: 88, 2025: 104, 2026: 107 so far. A few other
  IPOs may still be missing from NSE's list.

## Outside commentary
- Subscription (especially QIB) and grey-market premium are the factors most often tied to listing returns; see
  [Angel One](https://www.angelone.in/knowledge-center/ipo/ipo-oversubscription-versus-listing-gains) and
  [one blog analysis of 300 Indian IPOs](https://dev-ritik.github.io/blog/2025/12/31/i-analyzed-300-indian-ipos-heres-what-actually-works).
- 2025 weakness is attributed to aggressive pricing, heavy OFS, loss-making new-age issuers and supply:
  [Business Standard](https://www.business-standard.com/amp/markets/ipo/three-of-four-new-listings-flop-on-debut-in-2025-125093000900_1.html),
  [Business Standard](https://www.business-standard.com/amp/markets/news/ipos-in-h1-2025-were-priced-to-perfection-be-selective-now-say-analysts-125062600755_1.html).
- Listing gains do not always last:
  [Business Today](https://www.businesstoday.in/personal-finance/story/ipos-delivered-21-average-listing-gains-but-did-investors-actually-make-money-565507-2026-11-04).
