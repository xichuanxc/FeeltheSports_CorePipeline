# Video Table

*20 August 2026. Candidate counts at threshold 0.12, from `detection_stats.py`.
Full write-up with reasoning: `docs/VIDEO_INVENTORY.md`.*

| # | Video | Surface | Length | Candidates | Labels (hits) | Adjudicated | Role | Remark |
|---|---|---|---|---|---|---|---|---|
| 1 | Nick Kyrgios vs Roger Federer, Miami 2017 | hard | 38.6 min | 2,682 | 272 (147) | **10%** | **TEST** | Every accuracy figure rests on this file, and 90% of it was never judged. 2,410 candidates outstanding. Oldest broadcast, largest source of shoe squeaks. |
| 2 | Carlos Alcaraz v Novak Djokovic, AO 2026 Final | hard | 4.1 min | 158 | 154 (66) | 95% | **VALIDATION** | Picks the training epoch, so its numbers cannot be quoted as performance. Hardest video in the evaluation: the only one where gating does not reach zero false fires. |
| 3 | Hailey Baptiste vs Barbora Krejcikova, RG 2026 R1 | clay | 3.5 min | 191 | 301 (66) | 100% | train | Most heavily labelled file. Largest source of ambient noise (184) and speech (45). Heavy commentary, which is what makes it useful. |
| 4 | The Best of the World No.1! Sabalenka at Wimbledon 2025 | grass | 6.2 min | 182 | 185 (129) | 100% | train | Most racket hits of any fully adjudicated file. A montage, so rallies are cut tight and the crowd is near-continuous. |
| 5 | Benjamin Bonzi vs Alexander Zverev, RG 2026 R1 | clay | 3.0 min | 128 | 130 (45) | 100% | train | Quietest broadcast: speech covers 1.8%, and the Silero stage removes nothing here. Hardest for the classifier, lowest gated recall (0.69). |
| 6 | Perfect performance: Sabalenka v Osaka, Wimbledon 2026 | grass | 3.4 min | 112 | 112 (75) | 100% | train | Cleanest result anywhere: 0.9 false vibrations per minute ungated, 0.95 recall with zero false fires gated. |
| 7 | Aryna Sabalenka v Elena Rybakina, AO 2026 Final | hard | 3.6 min | 103 | 40 (23) | **39%** | train | **Cheapest unfinished work: about 63 candidates left.** Hard court, the surface furthest below target, and hard-court training data is thin because the other large hard file is the test video. |
| 8 | Maja Chwalinska vs Mirra Andreeva, RG 2026 Women's Final | clay | 12.4 min | 1,030 | 0 | **0%** | held out | Downloaded 31 July to test the trained model. Opened once, left clean. The only file that could support a genuinely unseen evaluation. |
| 9 | Incredible 13 MINUTE Game: Raducanu vs Sabalenka (In Full) | grass | 13.8 min | — | no CSV | — | unlabelled | Renamed 20 Aug to include *Wimbledon*, so the surface is now inferable and a CSV would bring it into the pipeline as grass. **The only continuous, uncut passage of play in the corpus**, everything else being highlights, which makes it the best available material for testing the live-broadcast limitation. |

## Corpus totals

| Surface | Samples | Share | Target |
|---|---|---|---|
| hard | 857 | 39.0% | 50% |
| clay | 792 | 36.1% | 35% |
| grass | 546 | 24.9% | 15% |

1,194 labelled events across seven files; 551 are racket hits.
