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

## How well it predicts (trained 2022-2025, tested on 105 IPOs from 2026)
| Method | Accuracy | Precision |
|---|---|---|
| Always apply | 69% | 69% |
| Total subscription over 10x | 73% | 88% |
| Logistic regression, all features | 70% | 76% |

Fundamentals-only models (what you have before bidding opens) reach AUC about 0.65 on 2026, only a little better than
a coin flip. Final subscription carries most of the signal, and it is only known when bidding closes.

## Outside commentary
- Subscription (especially QIB) and grey-market premium are the factors most often tied to listing returns; see
  [Angel One](https://www.angelone.in/knowledge-center/ipo/ipo-oversubscription-versus-listing-gains) and
  [one blog analysis of 300 Indian IPOs](https://dev-ritik.github.io/blog/2025/12/31/i-analyzed-300-indian-ipos-heres-what-actually-works).
- 2025 weakness is attributed to aggressive pricing, heavy OFS, loss-making new-age issuers and supply:
  [Business Standard](https://www.business-standard.com/amp/markets/ipo/three-of-four-new-listings-flop-on-debut-in-2025-125093000900_1.html),
  [Business Standard](https://www.business-standard.com/amp/markets/news/ipos-in-h1-2025-were-priced-to-perfection-be-selective-now-say-analysts-125062600755_1.html).
- Listing gains do not always last:
  [Business Today](https://www.businesstoday.in/personal-finance/story/ipos-delivered-21-average-listing-gains-but-did-investors-actually-make-money-565507-2026-11-04).
