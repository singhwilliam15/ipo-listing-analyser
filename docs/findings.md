# What separated IPOs that listed up from those that listed down

385 mainboard IPOs listed 2022 to 2026 (NSE data; fundamentals read from each RHP).
"Gain" means the listing-day open was above the issue price: 280 IPOs (average +26.7%).
"Loss" means flat or lower: 105 IPOs (average -5.7%).

| Feature | Gainers (median) | Losers (median) | Signal |
|---|---|---|---|
| QIB subscription | 90x | 4.2x | strong (Spearman 0.74) |
| Total subscription | 54x | 3.0x | strong (0.71) |
| Retail subscription | 13.8x | 2.3x | strong (0.59) |
| PAT margin | 8.9% | 6.0% | modest |
| Revenue CAGR | 27.9% | 21.2% | modest |
| Return on net worth | 22% | 18% | modest |
| Debt / equity | 0.46 | 0.74 | weak |
| P/E premium to peers | -12.7% | +1.0% | weak |
| Issue size, OFS share, promoter holding, anchor share, P/E | no difference | | none |

## Gainers
- QIB subscription of 50x or more: 95% gained (176 IPOs). Total subscription over 100x: all 62 gained, +49% on average.
- Debt/equity of 1 or less: 78-80% gained. P/E 20% or more below peers: 80%.
- IPO priced above the pre-IPO placement: 83% gained (53 IPOs).
- Strong sectors: internet and platforms, FMCG, healthcare, industrials, consumer (79-86%).

## Losers
- 53 of 105 losers were under 3x subscribed; 72 were under 10x. With QIB subscription under 5x only 41% gained.
- Debt/equity above 2: 49% gained. Loss-making companies: median listing gain +2.2% vs +10.5%.
- Weak sectors: banking and NBFC (44%), media and education (44%), logistics and travel (53%).
- Exceptions exist: Shree Ram Twistex (44x), Udayshivakumar Infra (31x) and R K Swamy (26x) were well subscribed and still lost.

## By year
| Year | IPOs | Listed above issue | Median gain |
|---|---|---|---|
| 2022 | 37 | 65% | +3.2% |
| 2023 | 56 | 80% | +16.5% |
| 2024 | 85 | 79% | +17.8% |
| 2025 | 100 | 70% | +6.2% |
| 2026 | 107 | 69% | +7.6% |

## Where this data disagrees with commentary
Analysts often blame OFS-heavy and aggressively priced issues, and small issues. In this sample OFS share and issue
size made no clear difference, and a high P/E alone did not predict a loss.

## Backtest: fitted on 2022-2025, scored on 2026

The dataset now has 394 mainboard IPOs. Five that NSE's past-issues list omits (Ola Electric, IREDA, NTPC Green,
Tata Capital, Aegis Vopak) were added from their NSE issue pages, DCX Systems and NSDL got their listing dates, and
IREDA's subscription was entered by hand from published figures. The winners-vs-losers tables above were computed on the
earlier 385-row version; the models below use the updated data. 285 IPOs listed in 2022-2025 are used for fitting and
107 listed in 2026 are used only for scoring. A test (`tests/test_report.py`) scrambles the 2026 outcomes and checks that no
fitted parameter or prediction changes.

Each score below is the model scored on the 2026 IPOs it never saw.

### Chance of a listing gain (P(open > issue price))

| Model | IPOs | AUC | Brier | Accuracy at 0.5 | Base rate |
|---|---|---|---|---|---|
| Subscription model (log QIB, total, retail subscription + PAT margin), final figures | 106 | **0.85** | 0.147 | **81%** | 69% |
| Fundamentals only (pre-bid) | 107 | 0.68 | 0.200 | 70% | 69% |

| Baseline | Accuracy |
|---|---|
| Always apply | 69% |
| Total subscription > 10x | 73% |

Calibration of the subscription model (deciles of predicted probability, about 10 IPOs each):

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Predicted | 32% | 39% | 45% | 54% | 71% | 84% | 90% | 93% | 94% | 95% |
| Actual | 27% | 45% | 10% | 73% | 80% | 82% | 80% | 91% | 100% | 100% |

It ranks well and the top deciles are close to calibrated. The middle is noisy (decile 3 came in at 10% vs 45% predicted)
because each decile holds only about 10 IPOs. The fundamentals-only model is overconfident at the bottom: its lowest decile
predicted 57% and 45% actually gained.

### Expected listing gain (%)

| Method | Median abs. error of the median | Covers 25-75 (target 50%) | Covers 10-90 (target 80%) | Spearman |
|---|---|---|---|---|
| Model A: quantile regression on QIB, total, retail, PAT margin, sector | **6.6 pts** | 56% | 76% | 0.77 |
| Model B: past IPOs with similar QIB subscription | 7.1 pts | 53% | 82% | 0.76 |
| Pre-bid: 10 most similar past IPOs | 12.2 pts | 35% | 64% | 0.24 |

Model A is the default. It was chosen on a 2025 validation fold inside the training data (pinball loss 5.04 vs 5.46 for B),
and 2026 agrees. Its ranges are a little narrow at the 10-90 end. Before bidding the 10-nearest-IPOs range is too narrow
to trust (covers 64% where 80% is intended) and barely correlated with outcomes, which is why the app labels it a weak
signal and shows it wide.

### Strategies on 2026

| Rule | IPOs applied | Hit rate | Avg listing gain | Allotment-weighted gain | Summed expected gain per lot |
|---|---|---|---|---|---|
| Always apply | 107 | 69% | +14.4% | +0.5% | ₹7,624 |
| Total subscription > 10x | 59 | 88% | +26.6% | +1.2% | ₹10,473 |
| QIB > 3x | 82 | 80% | +19.6% | +1.2% | ₹14,403 |
| QIB > 5x | 68 | 88% | +24.2% | +1.5% | **₹15,346** |
| QIB > 10x | 62 | 89% | +25.9% | +1.3% | ₹11,854 |
| QIB > 20x | 55 | 91% | +28.8% | +1.5% | ₹12,149 |
| QIB > 50x | 43 | 91% | +33.6% | +1.4% | ₹8,960 |

Allotment-weighted gain multiplies each IPO's gain by min(1, 1 / retail subscription), so heavily subscribed IPOs, where
the gain is large but allotment is rare, count for less. This is the practical lesson: a higher hit rate does not mean more
money per application. Summed over the year, QIB > 5x earned the most per lot (about ₹226 per IPO applied), more than
QIB > 50x, because the strongest IPOs are the hardest to get allotted.

### Caveats
- The 2026 sample is 107 IPOs, about 10 per calibration decile. Treat decile values as rough.
- The subscription model needs the final figures. At provisional stage the app shows its output as a floor, because QIBs bid
  mostly on the last day.
- The models are fitted on 2022-2025 only, so the scores above stay valid for what the app shows. The 2026 data is not fed
  back into them.
- Chittorgarh's mainboard list could not be fetched (its tables load through JavaScript), so completeness of the IPO list was
  checked only against NSE. Yearly mainboard counts in the data: 2022: 38, 2023: 57, 2024: 88, 2025: 104, 2026: 107 so far.
  A few other IPOs may still be missing from NSE's list.

## Outside commentary
- Subscription (especially QIB) and grey-market premium are the factors most often tied to listing returns; see
  [Angel One](https://www.angelone.in/knowledge-center/ipo/ipo-oversubscription-versus-listing-gains) and
  [one blog analysis of 300 Indian IPOs](https://dev-ritik.github.io/blog/2025/12/31/i-analyzed-300-indian-ipos-heres-what-actually-works).
- 2025 weakness is attributed to aggressive pricing, heavy OFS, loss-making new-age issuers and supply:
  [Business Standard](https://www.business-standard.com/amp/markets/ipo/three-of-four-new-listings-flop-on-debut-in-2025-125093000900_1.html),
  [Business Standard](https://www.business-standard.com/amp/markets/news/ipos-in-h1-2025-were-priced-to-perfection-be-selective-now-say-analysts-125062600755_1.html).
- Listing gains do not always last:
  [Business Today](https://www.businesstoday.in/personal-finance/story/ipos-delivered-21-average-listing-gains-but-did-investors-actually-make-money-565507-2026-11-04).
