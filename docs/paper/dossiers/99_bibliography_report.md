# 99 — Bibliography report (bibliography stage)

Prepared 2026-10-23 against the six dossiers in `docs/paper/dossiers/` (01–06) as they stood that day.
Output: `docs/paper/refs.bib` (124 entries). That file is outside `dossiers/`. The bibliography-stage
instruction names that path explicitly, so this stage writes there as a deliberate exception to the
general "dossiers only" rule. No other repository file was changed.

## 1. What was collected

- Every `[CITE?]` reference in the six dossiers. Sources: the reference lists (01 §13, 02 §18,
  03 §16, 04 §16, 05 §8, 06 "[CITE?] references") and inline `[CITE?]` mentions in the bodies.
- The core references named in the stage instruction, but only those that a dossier actually uses.
  Two of them are named in no dossier and are included because a dossier asks for the concept:
  DeMillo et al. (1978) and Jia & Harman (2011), for the mutation-testing prior-art question at
  01:374 and 01:562.
- 125 distinct items after de-duplication. 124 resolve to bib entries:
  - `stochvol` resolves to the two existing JSS entries;
  - Chan's toolkit gives two entries (MATLAB and Python);
  - the JSS author guidelines are excluded as a non-bibliographic planning reference.

## 2. How each item was verified (commands and pages actually used)

- **DOI items (96 lookups).** The Crossref REST record of each DOI was read with `curl
  https://api.crossref.org/works/<doi>`, run from a scratch script, output to stdout only. Fields
  read: authors, title, container, volume, issue, pages, print/online dates, publisher. Five items
  without a known DOI were found with `query.bibliographic`, and the returned title and first
  author were checked by eye. The `% verified:` line in `refs.bib` gives the resolvable
  `https://doi.org/<doi>` page.
- **Fields Crossref lacked** (pages, author lists). Taken from IDEAS/RePEc (Sims–Zha 1998,
  Newey–West 1987, FRED-QD), the Econometric Society page (Hansen–Lunde–Nason authors), NBER,
  PMLR, the Norges Bank page, and the CRAN citation files (JSS page ranges).
- **arXiv items.** Read with `curl https://export.arxiv.org/api/query?id_list=…` for 1804.06788,
  2310.14438, 2502.03279, 2001.06465, 1412.5218, 2002.09309 and 1912.11554.
- **R packages.** Read from the CRAN `index.html` and `citation.html` pages, the CRAN archive
  listings (stochvol 3.2.1, mfbvar 0.5.6, bayesianVARs), and GitHub DESCRIPTION files (mfbvar,
  bvars).
- **Python and other software.** PyPI JSON API (`bvar`, `scoringrules`, `impulso`,
  `srvar-toolkit`, `pymc-extras`, `jax`); GitHub raw README and citation blocks (JAX, scoringrules,
  BoE bvar); the GitHub API for the `joshuaccchan` and `ygeunkim` repositories; the PyMC-extras
  source (`class BayesianVARMAX`).
- **PDF metadata and title blocks.** Read for two PDFs: ECB WP 1934 (BEAR) and Frostig et al.
  (mlsys.org).
- **Holm (1979).** No registered DOI; checked through OpenAlex W2121044470 plus a secondary
  listing for the issue number.
- **Books.** The rule, applied uniformly:
  - a publisher or DOI book record where one exists: Higham, Björck, Chilès–Delfiner;
  - the Library of Congress MARC record (via Open Library) for Gupta–Nagar;
  - otherwise a DOI-registered book review that states author, title, publisher and year: Kish
    1965 and Journel–Huijbregts 1978.
- **Sites that could not be read.** These cases fell back to the sources above, and the table says
  so in each case:
  - `jstatsoft.org` refused connections (ECONNREFUSED), so JSS page ranges come from the packages'
    CRAN citation files;
  - `ams.org` shows a bot check to scripts;
  - JSTOR, Elgar and Inderscience returned 403.

## 3. Status definitions and counts

| status | meaning | count |
| :--- | :--- | ---: |
| verified | every detail the dossiers give matches the authoritative record | 83 |
| verified (completed) | details right but incomplete; missing fields (pages, DOI, venue, publisher, version) added from the record | 30 |
| **corrected** | at least one stated detail was wrong or unresolved: author list, year, venue, title or spelling (each change is listed) | 11 |
| NOT FOUND (excluded) | no authoritative record found | 0 |
| excluded | not a bibliography item (JSS author guidelines) | 1 |
| **total distinct items** | | **125** |

Not verified and not in `refs.bib`, as out of scope (no dossier cites them; SPEC §15 or task
list only):
- Chan, Eisenstat & Strachan (2020);
- Chipman, George & McCulloch (2010);
- Huber, Koop & Onorante (2021);
- Zhang, Naughton, Bondell & Reich (2022);
- Chan & Eisenstat (2018);
- Arias, Rubio-Ramírez & Waggoner (2018);
- Koop, Pesaran & Potter (1996);
- Waggoner & Zha (1999);
- Chan (2020, JBES; cited only in N08);
- Kastner, Frühwirth-Schnatter & Lopes (2017);
- Kim & Baek (2024, JSCS);
- Woźniak (2025, arXiv:2410.15090);
- "Chan & Grant" (task list, with a question mark; no dossier hook).

Add any of them if a later draft cites them.

## 4. The 11 corrections (field-level)

1. **Chan (2023) survey → `chan2024bvars`.**
   - Before: arXiv:2310.14438 (2023).
   - Published: chapter 3, pp. 43–67, in Clements & Galvão (eds.), *Handbook of Research Methods
     and Applications in Macroeconomic Forecasting*, Edward Elgar, 2024, DOI
     10.4337/9781035310050.00007.
   - The N08 section numbers refer to arXiv v1.
2. **Carriero, Clark, Marcellino & Mertens.** 03 §16 said "2022 or 2024": it is 2024, REStat
   106(5), 1403–1417.
3. **Yu & Meng (2011).** The title was truncated. The full title adds "—An
   Ancillarity–Sufficiency Interweaving Strategy (ASIS) for Boosting MCMC Efficiency".
4. **Bertsche & Braun.**
   - Before: 2020.
   - Now: JBES 40(1), 328–341, 2022. Online 2020; the SSRN WP is 2020.
5. **JSS 100(12) author list.** "Kastner (2021)" (N02, relayed in 03) → Hosszejni & Kastner.
6. **BGVAR.** "Böck" → "Boeck", as JSS, Crossref and CRAN spell it; key `boeck2022bgvar`.
7. **bayesianVARs authors.** Gruber, **Stefan** Haan & Kastner. 01 gave Haan the initial "P."
   and 04 dropped Haan.
8. **bvhar authors.** "Kim, Y. G." → Kim & Baek.
9. **bvars (R).**
   - Before: "Woźniak".
   - Now: Liu, Ramirez Hassan & Woźniak; on CRAN, version 1.1.
10. **scoringrules (Python).** "Zanetta & Allen?" resolved: Francesco Zanetta & Sam Allen, cited
    as 2024 per the README.
11. **JAX.** The author list has 12 names. 03 omitted Yash Katariya.

## 5. Editorial decisions left open

1. **FRED-QD year.**
   - 01/02 cite the *Review* 103(1), 1–44, January 2021: `mccracken2021fredqd`.
   - 06 and `src/rai_bvar/datasets/fred.py` say 2020, which is NBER WP 26872:
     `mccracken2020fredqd`.
   - Both are verified. The paper should cite the 2021 *Review* and align `fred.py` (02 D14).
2. **Matheron's rule source** (04 §16, 05 §8). Candidates, all verified:
   - `journel1978mining`;
   - `chiles2012geostatistics` (edition to choose);
   - for pathwise conditioning, `wilson2020efficiently`.
3. **Software versions.**
   - R packages are cited at the CRAN version current on 2026-10-23. The repo-used versions
     coincide where known: BVAR 1.0.5, bayesianVARs 0.1.8, bvartools 0.3.0.
   - stochvol 3.2.1, the version used, exists in the CRAN archive. It is cited through the JSS
     articles, as CRAN asks. The text should state the version.
   - Python: jax 0.12.0.1, scoringrules 0.11.0 and BoE bvar 0.3.5 are on PyPI.
4. **To check.** The Alfa–Xue–Ye page range: the AMS citation line says 217–236 and Crossref
   says 217–237.
5. **To check.** The Modrák et al. theorem and equation numbers cited from arXiv v3, against the
   journal version.
6. **mfbvar was archived on CRAN on 2022-08-08.** The landscape section must not present it as a
   current CRAN package.
7. **srvar-toolkit.** The first PyPI release is 2026-07-18. SPEC §1.4's "first release Aug 2026"
   should be rechecked; the landscape claims stay unverified (01 "Unsupported claims" item 4).
8. **To check at compile time.** `kadiyala1997numerical` has a SICI DOI containing `<` and `>`.
   Under OT1 font encoding these print as ¡ and ¿ inside `\texttt`; use `\usepackage[T1]{fontenc}`
   or check the PDF.
9. **Error in the stage instruction.** It describes the 2022 corrigendum as being "with Chan & Yu".
   The corrigendum's authors are Carriero, Chan, Clark & Marcellino; Yu co-authored Chan, Koop & Yu
   (2024). `refs.bib` follows the Crossref record. Do not copy the instruction's wording into the
   draft.

## 6. Scope caveat

This stage checked **bibliographic data only**. It does not verify any claim the dossiers make
about the content of these works. Examples:
- the third-party ordering magnitudes (01 unsupported item 7);
- "stochvol's JSS paper uses a Geweke variant in unit tests" (01 unsupported item 8);
- the Talts and Modrák theorem and section numbers;
- the CR 2015 table values.

Such claims keep the status the dossiers give them. No package claim changes status because a
reference was verified.

## 7. `[CITE?]` sites not matched by the regex mapping

I listed every `[CITE?]` line in the dossier bodies, outside the reference lists. Lines within two
lines of a mapped site were treated as covered, and the rest were read in context. Three are
meta-text: 01:6, 01:22 and 04:47. The others:

**Resolved by a bib key.** The name sits one or more lines above the flag:

| site | context | resolves to |
| :--- | :--- | :--- |
| 01:217 | stochvol/factorstochvol JSS article | `hosszejni2021modeling` |
| 01:341 | Gaussian-approximation log score | `adolfson2007forecasting` |
| 02:157 | corrected triangular algorithm | `carriero2022corrigendum` |
| 02:204 | Kronecker structure forces equal tightness | `kadiyala1997numerical` |
| 02:426 | structural form of the NIW prior, Prop. 1 / Cor. 1 | `chan2022asymmetric` |
| 02:818 | FRED-MD/QD transformation codes | `mccracken2016fredmd` (and `mccracken2021fredqd` for the QD codes) |
| 03:110 | corrected triangular algorithm | `carriero2022corrigendum` |
| 03:353 | partially collapsed Gibbs blocks | `vandyk2008partially` |
| 04:389 | Talts et al.'s objection to Geweke z-scores | `talts2018validating` |
| 04:434 | Gandy & Scott's inflated Geweke size | `gandy2021unit` |
| 04:607 | Modrák et al. Theorem 7 | `modrak2025simulation` |
| 04:884 | Geweke's "error 2" (mutant M3) | `geweke2004getting` |
| 05:580 | sign switch / measure-preserving fold | `fruhwirth2010stochastic` |
| 06:99 | CR (2015) publication | `clark2015macroeconomic` |
| 06:104 | CR (2012) working paper | `clark2012macroeconomic` |
| 06:116 | JPR (1994) single-move Metropolis step | `jacquier1994bayesian` |
| 06:126 | CCM (2016) +0.967 joint score | `carriero2016common` |
| 06:180 | CCCM (2022) triangular sampler | `carriero2022corrigendum` |

**Still open: no reference named in any dossier.** I did not invent one.

| site | what needs a citation | candidates (not assigned) |
| :--- | :--- | :--- |
| 01:590 | Q14: "NUTS mixes worse than precision-sampler Gibbs + ASIS on SV states" | None in the dossiers. The claim is unsupported (01 "Unsupported claims" item 2). Cite literature, benchmark, or drop the claim |
| 05:66 | "subtraction-free elimination" prior art for the excess-pivot recursion | Probably `grassmann1985regenerative`, `alfa2002accurate`, `ye2008computing` (05:328–329 and 05:881 position F2 against GTH/Ye). The authors must confirm |
| 05:215 (third `[CITE?]` after "Higham; Björck;") | a further textbook for normal equations vs QR conditioning | `higham2002accuracy` and `bjorck1996numerical` are already named; the extra slot is open |
| 05:216 | "QR-based Gaussian conditional draws are standard in Bayesian regression" | No reference proposed in any dossier: open |

## 8. Per-reference table

Notes on the columns:
- **As stated in the dossiers.** My transcription of the dossiers' wording, with diacritics
  dropped.
- **Sites.** `NN:LLL` = dossier `NN_*.md`, line `LLL`. A trailing `r` marks an entry in that
  dossier's `[CITE?]` reference list; no suffix means an inline `[CITE?]` mention on that line or
  the next. Line numbers are as of 2026-10-23. Sites were found by regex over the dossiers, so
  read them as pointers, not as an exhaustive concordance.

### BVAR priors and mean specification

| bib key | as stated in the dossiers | status | what changed / notes | [CITE?] sites |
| :--- | :--- | :--- | :--- | :--- |
| `litterman1986forecasting` | Litterman (1986). Forecasting with Bayesian vector autoregressions: five years of experience. JBES 4(1), 25-38. | verified | Title on the JBES record: 'Forecasting With Bayesian Vector Autoregressions—Five Years of Experience'. | 01:629r 01:630r 02:1690r 02:1706r |
| `doan1984forecasting` | Doan, Litterman & Sims (1984). Forecasting and conditional projection using realistic prior distributions. Econometric Reviews 3(1), 1-100. | verified | — | 01:630r 02:1690r |
| `sims1998bayesian` | Sims & Zha (1998). Bayesian methods for dynamic multivariate models. IER 39(4), 949-968. | verified | Crossref gives first page only; 949–968 from IDEAS/RePEc. | 02:1714r |
| `sims2006were` | Sims & Zha (2006). Were there regime switches in U.S. monetary policy? AER 96(1), 54-81. | verified | — | 01:43 01:632r 02:1715r |
| `banbura2010large` | Banbura, Giannone & Reichlin (2010). Large Bayesian vector auto regressions. JAE 25(1), 71-92. | verified | — | 02:1677r |
| `giannone2015prior` | Giannone, Lenza & Primiceri (2015). Prior selection for vector autoregressions. REStat 97(2), 436-451. | verified | — | 01:626r 02:1692r 04:843 04:1191r 05:960r 05:961r |
| `lenza2022estimate` | Lenza & Primiceri (2022). How to estimate a vector autoregression after March 2020. JAE 37(4), 688-699. | verified | — | 01:627r 02:1705r |
| `kadiyala1997numerical` | Kadiyala & Karlsson (1997). Numerical methods for estimation and inference in Bayesian VAR-models. JAE 12(2), 99-132. | verified | — | 01:628r 02:203 02:1701r 06:113 06:952r |
| `chan2022asymmetric` | Chan (2022). Asymmetric conjugate priors for large Bayesian VARs. Quantitative Economics 13(3), 1145-1169. | verified | — | 01:108 01:640r 02:425 02:1683r 03:720 03:1560r 04:713 04:1174r 06:186 06:936r |
| `chan2024bvars` | Chan (2023). BVARs and stochastic volatility (survey; arXiv:2310.14438). | **corrected** | Dossier: Chan (2023), arXiv:2310.14438 survey. Published version: Chapter 3, pp. 43–67, in Clements & Galvão (eds.), *Handbook of Research Methods and Applications in Macroeconomic Forecasting*, Edward Elgar, 2024, DOI 10.4337/9781035310050.00007. Key renamed chan2024bvars. The section numbers quoted in N08 (§2.1, §3.1, §4.2.1, §4.2.3) refer to arXiv v1 and may differ in the chapter. | 01:641r |
| `carriero2015bayesian` | Carriero, Clark & Marcellino (2015). Bayesian VARs: specification choices and forecast accuracy. JAE. | verified (completed) | Added JAE 30(1), 46–73, DOI 10.1002/jae.2315 (online 2013, issue 2015). | 01:561 01:634r |
| `carriero2016common` | Carriero, Clark & Marcellino (2016). Common drifting volatility in large Bayesian VARs. JBES 34(3), 375-390. | verified | — | 01:635r 06:938r |
| `carriero2019large` | Carriero, Clark & Marcellino (2019). Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors. J. Econometrics 212(1), 137-154. | verified | — | 01:62 01:636r 02:1681r 03:1537r 04:1168r 05:944r 06:943r |
| `bognanni2022comment` | Bognanni (2022). Comment on 'Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors'. J. Econometrics 227(2), 498-505. | verified | — | 01:63 01:637r 02:155 02:1678r 04:1172r |
| `carriero2022corrigendum` | Carriero, Chan, Clark & Marcellino (2022). Corrigendum to 'Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors'. J. Econometrics 227(2), 506-512. | verified | Official title ends with '[J. Econometrics 212 (1) (2019) 137–154]'; the record gives the second author as 'Chan, Joshua'. | 01:64 01:109 01:638r 02:156 02:1680r 03:1534r 04:1170r 05:947r 06:940r |
| `carriero2024addressing` | Carriero, Clark, Marcellino & Mertens (2024; one dossier: '2022 or 2024'). Addressing COVID-19 outliers in BVARs with stochastic volatility. REStat 106(5), 1403-1417. | **corrected** | 03 §16 item 17 said '2022 or 2024 … year and venue to verify': it is REStat 106(5), 1403–1417, 2024, DOI 10.1162/rest_a_01213 (01, 02 were already right). | 01:639r 02:608 02:1682r 03:1574r |
| `arias2023macroeconomic` | Arias, Rubio-Ramirez & Shin (2023). Macroeconomic forecasting and variable ordering in multivariate stochastic volatility models. J. Econometrics 235(2), 1054-1086. | verified | — | 01:68 01:378 01:642r 02:1676r 06:517 06:946r |
| `chan2024large` | Chan, Koop & Yu (2024). Large order-invariant Bayesian VARs with stochastic volatility. JBES 42(2), 825-837. | verified | Issue year 2024; published online 2023-09-28. | 01:68 01:378 01:643r 02:402 02:1685r 06:949r |
| `primiceri2005time` | Primiceri (2005). Time varying structural vector autoregressions and monetary policy. REStud 72(3), 821-852. | verified | — | 01:644r 02:1712r 04:883 04:1185r |
| `delnegro2015time` | Del Negro & Primiceri (2015). Time varying structural vector autoregressions and monetary policy: a corrigendum. REStud 82(4), 1342-1345. | verified | — | 01:432 01:645r 02:654 02:1688r 03:348 03:1540r 04:1183r |
| `cogley2005drifts` | Cogley & Sargent (2005). Drifts and volatilities: monetary policies and outcomes in the post WWII US. RED 8(2), 262-302. | verified | — | 01:646r 06:959r |
| `kastner2019sparse` | Kastner (2019). Sparse Bayesian time-varying covariance estimation in many dimensions. J. Econometrics 210(1). | verified (completed) | Added pages 98–115. | 01:603r |
| `bertsche2022identification` | Bertsche & Braun (2020). Identification of structural vector autoregressions by stochastic volatility. | **corrected** | Year 2020 → 2022. It is JBES 40(1), 328–341, DOI 10.1080/07350015.2020.1813588 (online 2020-10-12). 2020 is the year of the SSRN working paper, DOI 10.2139/ssrn.3627718. | 01:657r |
| `huber2022inference` | Huber & Rossini (2022). Inference in Bayesian additive vector autoregressive tree models. AoAS 16(1). | verified (completed) | Added pages 104–123. | 01:194 01:654r |
| `clark2023tail` | Clark, Huber, Koop, Marcellino & Pfarrhofer (2023). Tail forecasting with multivariate Bayesian additive regression trees. IER 64(3). | verified (completed) | Added pages 979–1022. | 01:655r |
| `gruber2025forecasting` | Gruber & Kastner (2025). Forecasting macroeconomic data with Bayesian VARs: sparse or dense? It depends! IJF 41(4). | verified (completed) | Added pages 1589–1619 and DOI. | 01:656r |

### Stochastic volatility, state-space samplers, random variates

| bib key | as stated in the dossiers | status | what changed / notes | [CITE?] sites |
| :--- | :--- | :--- | :--- | :--- |
| `kim1998stochastic` | Kim, Shephard & Chib (1998). Stochastic volatility: likelihood inference and comparison with ARCH models. REStud 65(3), 361-393. | verified | Crossref record misspells the second author as 'Shepherd' and gives 'Sangjoon'. The bib uses Shephard: that is the spelling in the Omori et al. (2007) record and in the CRAN citation files. | 01:647r 02:1703r 03:1545r 04:1176r |
| `omori2007stochastic` | Omori, Chib, Shephard & Nakajima (2007). Stochastic volatility with leverage: fast and efficient likelihood inference. J. Econometrics 140(2), 425-449. | verified | — | 01:648r 02:587 02:1711r 03:1543r 04:1178r 05:977r |
| `kastner2014ancillarity` | Kastner & Fruhwirth-Schnatter (2014). Ancillarity-sufficiency interweaving strategy (ASIS) for boosting MCMC estimation of stochastic volatility models. CSDA 76, 408-423. | verified | — | 01:432 01:649r 02:593 02:1702r 03:1547r 05:970r 06:985r |
| `yu2011center` | Yu & Meng (2011). To center or not to center: that is not the question. JCGS 20(3), 531-570. | **corrected** | Title was truncated. Full title: 'To Center or Not to Center: That Is Not the Question—An Ancillarity–Sufficiency Interweaving Strategy (ASIS) for Boosting MCMC Efficiency'. | 05:991r |
| `fruhwirth2010stochastic` | Fruhwirth-Schnatter & Wagner (2010). Stochastic model specification search for Gaussian and partial non-Gaussian state space models. J. Econometrics 154(1), 85-100. | verified | — | 03:1599r 05:955r |
| `chan2009efficient` | Chan & Jeliazkov (2009). Efficient simulation and integrated likelihood estimation in state space models. IJMMNO 1(1/2), 101-120. | verified | Crossref gives first page only. 101–120 comes from the Inderscience listing and the ANU repository record, both seen only through web-search results (Inderscience returns 403 to scripts). | 01:650r 02:1684r 03:1552r 05:950r 06:988r |
| `mccausland2011simulation` | McCausland, Miller & Pelletier (2011). Simulation smoothing for state-space models: a computational efficiency analysis. CSDA 55(1), 199-212. | verified | — | 01:651r 02:1707r 03:1555r 05:975r 06:992r |
| `rue2001fast` | Rue (2001). Fast sampling of Gaussian Markov random fields. JRSS-B 63(2), 325-338. | verified | 03 marked it 'from memory'. All details are correct. | 03:1558r 05:980r |
| `stroud2003nonlinear` | Stroud, Muller & Polson (2003). Nonlinear state-space models with state-dependent variances. JASA 98(462), 377-386. | verified (completed) | Added DOI 10.1198/016214503000161. 98(462), 377–386 confirmed. | 01:432 01:652r 03:1583r 04:1187r |
| `jacquier1994bayesian` | Jacquier, Polson & Rossi (1994). Bayesian analysis of stochastic volatility models. JBES 12(4), 371-389. | verified | — | 06:954r |
| `jacquier2004bayesian` | Jacquier, Polson & Rossi (2004). Bayesian analysis of stochastic volatility models with fat-tails and correlated errors. J. Econometrics 122(1), 185-212. | verified | — | 01:653r 02:617 02:1700r 03:1571r 06:956r |
| `stock2016core` | Stock & Watson (2016). Core inflation and trend inflation. REStat 98(4), 770-784. | verified | — | 02:1716r |
| `vandyk2008partially` | van Dyk & Park (2008). Partially collapsed Gibbs samplers: theory and methods. JASA 103(482), 790-796. | verified | 03 marked it 'from memory'. All details are correct. | 02:1718r 03:352 03:1567r |
| `fox2017accelerated` | Fox & Parker (2017). Accelerated Gibbs sampling of normal distributions using matrix splittings and polynomials. Bernoulli 23(4B), 3711-3743. | verified | — | 03:543 03:1569r |
| `marsaglia2000simple` | Marsaglia & Tsang (2000). A simple method for generating gamma variables. ACM TOMS 26(3), 363-372. | verified | — | 03:734 03:1562r |
| `salmon2011parallel` | Salmon, Moraes, Dror & Shaw (2011). Parallel random numbers: as easy as 1, 2, 3. Proceedings of SC11. | verified (completed) | 'Proceedings of SC11' (from memory) completed: *Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis (SC '11)*, ACM, pp. 1–12, DOI 10.1145/2063384.2063405. | 03:1564r 03:1565r |
| `clark2011realtime` | Clark (2011). Real-time density forecasts from Bayesian vector autoregressions with stochastic volatility. JBES 29(3), 327-341. | verified | — | 06:995r |

### Clark–Ravazzolo target, real-time data, FRED

| bib key | as stated in the dossiers | status | what changed / notes | [CITE?] sites |
| :--- | :--- | :--- | :--- | :--- |
| `clark2015macroeconomic` | Clark & Ravazzolo (2015). Macroeconomic forecasting performance under alternative specifications of time-varying volatility. JAE 30(4), 551-575. | verified | 06 gave the title 'from memory'; it is correct. Issue 2015, online 2014-01-21. | 01:43 01:633r 02:1687r 03:1592r 04:1189r 05:952r 06:98 06:927r |
| `clark2012macroeconomic` | Clark & Ravazzolo (2012). The macroeconomic forecasting performance of autoregressive models with alternative specifications of time-varying volatility. FRB Cleveland WP 12-18 / Norges Bank WP 2012/09. | verified (completed) | Added DOI 10.26509/frbc-wp-201218. Norges Bank WP 9/2012 confirmed on the Norges Bank page. | 06:933r |
| `croushore2001realtime` | Croushore & Stark (2001). A real-time data set for macroeconomists. J. Econometrics 105(1), 111-130. | verified | — | 06:983r |
| `mccracken2016fredmd` | McCracken & Ng (2016). FRED-MD: a monthly database for macroeconomic research. JBES 34(4), 574-589. | verified | — | 01:289 01:684r 02:1708r 06:979r |
| `mccracken2021fredqd` | McCracken & Ng (2021). FRED-QD: a quarterly database for macroeconomic research. FRB St. Louis Review 103(1), 1-44 (06: 2020 NBER WP 26872; fred.py says 2020). | verified | Two versions exist: FRB St. Louis *Review* 103(1), 1–44, January 2021, DOI 10.20955/r.103.1-44 (01, 02 right), and NBER WP 26872, March 2020 (06 item 23). `fred.py`'s '2020' refers to the WP. Decide which one the paper cites (02 D14). | 01:289 01:685r 02:1709r 06:981r |
| `mccracken2020fredqd` | McCracken & Ng (2020). FRED-QD: a quarterly database for macroeconomic research. NBER WP 26872. | verified | NBER WP 26872, March 2020, DOI 10.3386/w26872. The NBER page links the 2021 Review version. | 06:982r |

### Matrix distributions, HAC

| bib key | as stated in the dossiers | status | what changed / notes | [CITE?] sites |
| :--- | :--- | :--- | :--- | :--- |
| `gupta2000matrix` | Gupta & Nagar (2000). Matrix Variate Distributions. Chapman & Hall/CRC. | verified | Chapman & Hall/CRC, Boca Raton, series vol. 104, ISBN 1-58488-046-5, from the Library of Congress MARC record (Open Library OL16983496M; imprint 2000). Publisher listings shown by web search date it Oct 1999. A 2018 e-book reissue has DOI 10.1201/9780203749289. | 02:1695r |
| `newey1987simple` | Newey & West (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. Econometrica 55(3), 703-708. | verified | Crossref gives first page only; 703–708 from IDEAS/RePEc. | 02:1710r |

### Forecast evaluation

| bib key | as stated in the dossiers | status | what changed / notes | [CITE?] sites |
| :--- | :--- | :--- | :--- | :--- |
| `diebold1995comparing` | Diebold & Mariano (1995). Comparing predictive accuracy. JBES 13(3), 253-263. | verified | — | 01:660r 02:1689r 06:242 06:961r |
| `harvey1997testing` | Harvey, Leybourne & Newbold (1997). Testing the equality of prediction mean squared errors. IJF 13(2), 281-291. | verified | — | 01:661r 02:1697r 06:243 06:963r |
| `clark2001tests` | Clark & McCracken (2001). Tests of equal forecast accuracy and encompassing for nested models. J. Econometrics 105(1), 85-110. | verified | — | 01:60 01:662r 02:1155 02:1686r 06:245 06:967r |
| `giacomini2006tests` | Giacomini & White (2006). Tests of conditional predictive ability. Econometrica 74(6), 1545-1578. | verified | — | 01:60 01:663r 02:1155 02:1691r 06:246 06:969r |
| `amisano2007comparing` | Amisano & Giacomini (2007). Comparing density forecasts via weighted likelihood ratio tests. JBES 25(2), 177-190. | verified | — | 01:664r 02:1675r |
| `hansen2011model` | Hansen, Lunde & Nason (2011). The model confidence set. Econometrica 79(2), 453-497. | verified | The Crossref record has no author list; authors from the Econometric Society page. | 01:354 01:469 01:665r 02:1159 02:1696r 06:252 06:965r |
| `gneiting2007strictly` | Gneiting & Raftery (2007). Strictly proper scoring rules, prediction, and estimation. JASA 102(477), 359-378. | verified | — | 01:469 01:666r 02:1693r 06:228 06:971r |
| `gneiting2011comparing` | Gneiting & Ranjan (2011). Comparing density forecasts using threshold- and quantile-weighted scoring rules. JBES 29(3), 411-422. | verified | — | 01:470 01:667r 02:1129 02:1694r |
| `scheuerer2015variogram` | Scheuerer & Hamill (2015). Variogram-based proper scoring rules for probabilistic forecasts of multivariate quantities. MWR 143(4), 1321-1334. | verified | — | 01:470 01:668r 02:1713r |
| `geweke2010comparing` | Geweke & Amisano (2010). Comparing and evaluating Bayesian predictive distributions of asset returns. IJF. | verified (completed) | Added IJF 26(2), 216–230, DOI. | 01:340 01:669r |
| `adolfson2007forecasting` | Adolfson, Linde & Villani (2007). Forecasting performance of an open economy DSGE model. Econometric Reviews 26(2-4), 289-328. | verified | — | 01:340 01:670r 02:1674r 06:225 06:934r |
| `burkner2020approximate` | Burkner, Gabry & Vehtari (2020). Approximate leave-future-out cross-validation for Bayesian time series models. JSCS 90(14). | verified (completed) | Added pages 2499–2523 (01 gave 90(14) only). | 01:307 01:308 01:671r |
| `kruger2021predictive` | Kruger, Lerch, Thorarinsdottir & Gneiting (2021). Predictive inference based on Markov chain Monte Carlo output. ISR 89(2), 274-301. | verified | Issue 2021, online 2020. | 06:228 06:973r |
| `dawid1999coherent` | Dawid & Sebastiani (1999). Coherent dispersion criteria for optimal experimental design. AoS 27(1), 65-81. | verified | — | 06:998r |
| `holm1979simple` | Holm (1979). A simple sequentially rejective multiple test procedure. Scand. J. Statistics. | verified (completed) | Added author's first name (Sture) and 6(2), 65–70. No registered DOI: the 10.2307/4615733 that OpenAlex gives does not resolve (doi.org 404), so the bib carries the JSTOR stable URL. | 04:1165r |

### Sampler validation, MCMC diagnostics, software testing

| bib key | as stated in the dossiers | status | what changed / notes | [CITE?] sites |
| :--- | :--- | :--- | :--- | :--- |
| `geweke2004getting` | Geweke (2004). Getting it right: joint distribution tests of posterior simulators. JASA 99(467), 799-804. | verified | — | 01:361 01:674r 03:1577r 04:407 04:1140r 05:958r |
| `talts2018validating` | Talts, Betancourt, Simpson, Vehtari & Gelman (2018). Validating Bayesian inference algorithms with simulation-based calibration. arXiv:1804.06788. | verified | Still arXiv only; v2 is dated 2020-10-21. Cited as 2018. | 01:361 01:675r 04:388 04:1142r |
| `modrak2025simulation` | Modrak, Moon, Kim, Burkner, Huurre, Faltejskova, Gelman & Vehtari (2025). Simulation-based calibration checking for Bayesian computation: the choice of test quantities shapes sensitivity. Bayesian Analysis 20(2), 461-488. | verified | Crossref truncates Bürkner's given name to 'Paul'. Theorem and equation numbers in the dossiers (e.g. Theorem 7, eq. 7) come from arXiv v3 (N06 open item 9). Check them against the journal version before quoting. | 01:361 01:676r 04:613 04:1144r |
| `sailynoja2022graphical` | Sailynoja, Burkner & Vehtari (2022). Graphical test for discrete uniformity and its applications in goodness-of-fit evaluation and multiple sample comparison. Statistics and Computing 32, article 32. | verified | 32(2), article 32. | 01:677r 04:613 04:1147r |
| `sailynoja2025posterior` | Sailynoja, Schmitt, Burkner & Vehtari (2025). Posterior SBC: simulation-based calibration checking conditional on data. arXiv:2502.03279. | verified | arXiv v2 (2025-03-10); no journal version found. | 04:1150r 04:1151r |
| `gandy2021unit` | Gandy & Scott (2021; arXiv v2). Unit testing for MCMC and other Monte Carlo methods. arXiv:2001.06465. | verified | No journal version found by search; arXiv v2 dated 2021-09-19. Companion CRAN package mcunit. | 01:362 01:679r 04:433 04:468 04:585 04:1152r |
| `grosse2014testing` | Grosse & Duvenaud (2014). Testing MCMC code. arXiv:1412.5218. | verified | arXiv v1. Presented at the 2014 NIPS workshop on Software Engineering for Machine Learning. | 01:361 01:678r 03:817 03:1579r 04:309 04:1154r |
| `cook2006validation` | Cook, Gelman & Rubin (2006). Validation of software for Bayesian models using posterior quantiles. JCGS 15(3), 675-692. | verified | — | 04:1155r |
| `gelman2017correction` | Gelman (2017). Correction to Cook, Gelman, and Rubin (2006). JCGS 26(4), 940. | verified | — | 04:1156r |
| `besag1989generalized` | Besag & Clifford (1989). Generalized Monte Carlo significance tests. Biometrika 76(4), 633-642. | verified | — | 04:1157r |
| `vehtari2021rank` | Vehtari, Gelman, Simpson, Carpenter & Burkner (2021). Rank-normalization, folding, and localization: an improved R-hat for assessing convergence of MCMC. Bayesian Analysis 16(2), 667-718. | verified | The official title ends '(with Discussion)'; kept in the bib. | 01:680r 02:1719r 03:1580r 04:1003 04:1159r 05:982r 06:259 06:976r |
| `geyer1992practical` | Geyer (1992). Practical Markov chain Monte Carlo. Statistical Science 7(4), 473-483. | verified | 473–483 is the article; 473–511 includes the discussion (N06 §13). | 04:586 04:1162r |
| `flegal2010batch` | Flegal & Jones (2010). Batch means and spectral variance estimators in Markov chain Monte Carlo. AoS 38(2), 1034-1070. | verified | — | 04:1163r |
| `kish1965survey` | Kish (1965). Survey Sampling. | verified (completed) | Added publisher (John Wiley & Sons, New York). Verified from a DOI-registered book review (APSR 59(4), 1965). No publisher page for the 1965 edition. | 04:1167r |
| `dutta2018testing` | Dutta, Legunsen, Huang & Misailovic (2018). Testing probabilistic programming systems. ESEC/FSE. | verified (completed) | Added pages 574–586, DOI. | 01:375 01:681r |
| `demillo1978hints` | (core list) DeMillo, Lipton & Sayward (1978). Hints on test data selection: help for the practicing programmer. Computer 11(4). | verified | Core list item (not named in any dossier). Resolves the mutation-testing prior-art requests at 01:374 and 01:562. | 01:374 01:562 01:681r |
| `jia2011analysis` | (core list) Jia & Harman (2011). An analysis and survey of the development of mutation testing. IEEE TSE 37(5). | verified | Core list item (not named in any dossier). Same sites as DeMillo et al. | 01:374 01:562 01:681r |

### Numerical linear algebra, Gaussian conditioning

| bib key | as stated in the dossiers | status | what changed / notes | [CITE?] sites |
| :--- | :--- | :--- | :--- | :--- |
| `alfa2002accurate` | Alfa, Xue & Ye (2002). Accurate computation of the smallest eigenvalue of a diagonally dominant M-matrix. Math. Comp. 71(237), 217-236. | verified | The AMS citation line says 217–236 and the bib follows it. The Crossref deposit says 217–237 and gives 2001-05-14 (electronic) as the issue date. **to check**: open the AMS page (it shows a bot check to scripts). | 05:329 05:941r |
| `ye2008computing` | Ye (2008). Computing singular values of diagonally dominant matrices to high relative accuracy. Math. Comp. 77(264), 2195-2230. | verified | — | 05:329 05:881 05:989r |
| `grassmann1985regenerative` | Grassmann, Taksar & Heyman (1985). Regenerative analysis and steady state distributions for Markov chains. Operations Research 33(5), 1107-1116. | verified | — | 05:328 05:881 05:963r |
| `higham2002accuracy` | Higham (2002). Accuracy and Stability of Numerical Algorithms (2nd ed.). SIAM. | verified | 2nd ed., SIAM, DOI 10.1137/1.9780898718027. | 05:214 05:965r |
| `bjorck1996numerical` | Bjorck (1996). Numerical Methods for Least Squares Problems. SIAM. | verified | SIAM, DOI 10.1137/1.9781611971484. | 05:215 05:943r |
| `wilson2020efficiently` | Wilson, Borovitskiy, Terenin, Mostowsky & Deisenroth (2020). Efficiently sampling functions from Gaussian process posteriors. ICML 2020. | verified (completed) | 'ICML 2020' completed: PMLR vol. 119, pp. 10292–10302, eds. Daumé III & Singh. | 04:1203r 05:985r |
| `journel1978mining` | Journel & Huijbregts (1978). Mining Geostatistics. Academic Press. | verified (completed) | Added publisher (Academic Press, London). Verified from a DOI-registered review in *Mineralogical Magazine* 43(328), 1979. | 04:1202r 04:1203r 05:476 05:986r 05:987r |
| `chiles2012geostatistics` | Chiles & Delfiner. Geostatistics (edition/year not given). | verified (completed) | The dossier gave no edition or year. The bib uses the 2nd ed. (Wiley, 2012, DOI 10.1002/9781118136188); Matheron's rule appears in both editions (1999, 2012). Which edition to cite is an editorial choice. | 04:1202r 05:476 05:986r 05:987r 05:988r |

### Software: JSS articles

| bib key | as stated in the dossiers | status | what changed / notes | [CITE?] sites |
| :--- | :--- | :--- | :--- | :--- |
| `kuschnig2021bvar` | Kuschnig & Vashold (2021). BVAR: Bayesian vector autoregressions with hierarchical prior selection in R. JSS 100(14), 1-27. | verified | Pages 1–27 are from the CRAN citation file; jstatsoft.org refused connections during this stage. | 01:186 01:214 01:494 01:600r 02:1704r 04:832 04:1193r 05:973r |
| `hosszejni2021modeling` | Hosszejni & Kastner (2021) [N02: 'Kastner (2021)']. Modeling univariate and multivariate stochastic volatility in R with stochvol and factorstochvol. JSS 100(12). | **corrected** | N02 (relayed in 03 §16 item 25) cites 'Kastner (2021)' for this article. The authors are Hosszejni & Kastner. Pages 1–34 added (CRAN citation file). | 01:187 01:216 01:601r 02:1698r 03:1595r 03:1597r 04:1195r 05:966r |
| `kastner2016dealing` | Kastner (2016). Dealing with stochastic volatility in time series using the R package stochvol. JSS 69(5). | verified (completed) | Added pages 1–30 (CRAN stochvol citation file). | 03:1550r 05:968r |
| `boeck2022bgvar` | Bock, Feldkircher & Huber (2022). BGVAR: Bayesian global vector autoregressions with shrinkage priors in R. JSS 104(9). | **corrected** | 'Böck' → 'Boeck', as JSS, Crossref and CRAN spell it. Key changed to boeck2022bgvar. Pages 1–28 added (CRAN citation file). | 01:221 01:613r |
| `jordan2019evaluating` | Jordan, Kruger & Lerch (2019). Evaluating probabilistic forecasts with scoringRules. JSS 90(12). | verified (completed) | Added pages 1–37 (CRAN citation file). | 01:222 01:223 01:615r |

### Software: R packages

| bib key | as stated in the dossiers | status | what changed / notes | [CITE?] sites |
| :--- | :--- | :--- | :--- | :--- |
| `gruber2026bayesianvars` | Gruber, Haan & Kastner / Gruber & Kastner: bayesianVARs R package (CRAN). | **corrected** | 01 lists 'Gruber, L., Haan, P. & Kastner, G.' and 04 lists 'Gruber & Kastner'. CRAN lists Luis Gruber, Stefan Haan, Gregor Kastner. Version 0.1.8 (2026-02-19) is the version the repo failed to build. CRAN asks that the methodology be cited as gruber2025forecasting. | 01:188 01:192 01:204 01:602r 04:1198r |
| `kastner2026factorstochvol` | Kastner: factorstochvol R package. | verified (completed) | Kastner (aut, cre), version 1.1.2 (2026-03-30). CRAN asks users to cite hosszejni2021modeling. | 01:189 01:603r |
| `wozniak2026bsvars` | Wozniak: bsvars R package. | verified (completed) | Woźniak, version 4.0 (2026-08-22). The CRAN citation file also lists a WP, arXiv:2410.15090. It has not been checked separately. | 01:193 01:222 01:604r |
| `wang2026bsvarsigns` | Wang & Wozniak: bsvarSIGNs R package. | verified (completed) | Wang & Woźniak, version 3.0 (2026-08-21). The citation file still says version 2.0 / 2025; the bib follows the CRAN page. | 01:193 01:604r |
| `krueger2015bvarsv` | Kruger: bvarsv R package. | verified (completed) | Fabian Krueger (spelled this way on CRAN), version 1.1 (2015-11-25). | 01:191 01:605r |
| `mohr2026bvartools` | Mohr: bvartools R package. | verified | Franz X. Mohr, version 0.3.0 (2026-09-11). 01:192 says the repo opened bvartools 0.3.0. | 01:192 01:606r |
| `kim2026bvhar` | Kim, Y. G.: bvhar R package (and Python port). | **corrected** | 'Kim, Y. G.' → Kim & Baek (Young Geun Kim, Changryong Baek), version 2.4.1 (2026-06-05). The GitHub repo ygeunkim/bvhar describes itself as an 'R and Python package'. The JSCS paper Kim & Baek (2024) is not in the bib. | 01:609r |
| `ankargren2021mfbvar` | Ankargren & Yang: mfbvar, mixed-frequency Bayesian VAR models in R. | verified (completed) | Sebastian Ankargren and Yukai Yang (Gregor Kastner, ctb), last version 0.5.6 (2021). **Archived on CRAN 2022-08-08**. The paper must not describe it as a current CRAN package. | 01:222 01:614r |
| `hosszejni2021modeling, kastner2016dealing` | stochvol R package (oracle; version 3.2.1 used). | verified | No separate @Manual: stochvol's CRAN citation file asks users to cite the two JSS articles. Version 3.2.1, which the parity runs used, is in the CRAN archive (2023-03-10). The current version is 3.2.9, which DESIGN §9.9 names. | 01:187 01:192 |
| `kuschnig2024bvar` | BVAR R package (oracle; version 1.0.5 used). | verified | CRAN citation file: 'BVAR: Hierarchical Prior Selection in R', R package version 1.0.5 (2024). That matches the version the parity fixtures used (01:186). | 01:186 04:832 |
| `liu2026bvars` | Wozniak: bvars R package (documentation: bsvars.r-universe.dev). | **corrected** | 'Woźniak, T. — bvars' → authors Rui Liu, Andrés Ramirez Hassan, Tomasz Woźniak. It is on CRAN, version 1.1 (2026-09-07), DOI 10.32614/CRAN.package.bvars. | 01:190 01:604r |

### Software: Python, MATLAB, other

| bib key | as stated in the dossiers | status | what changed / notes | [CITE?] sites |
| :--- | :--- | :--- | :--- | :--- |
| `labonne2026bvar` | Bank of England bvar Python package (PyPI v0.3.5; github.com/bank-of-england/bvar). | verified (completed) | No CITATION file and no preferred citation in the repo README. The PyPI metadata lists the authors as Paul Labonne, Andrea Renzetti and Joseph Oyegoke (Bank of England). Version 0.3.5 (2026-09-25), MIT. The bib cites it as an @Manual with organization 'Bank of England'. | 01:607r 04:1199r 05:993r |
| `chan2026bvartoolkit, chan2026bvartoolkitpython` | Chan, J. C. C.: BVAR toolkit (MATLAB) and bvar-toolkit-python. | verified | Both repos exist: joshuaccchan/bvar-toolkit (MATLAB, created 2026-09-01; linked from joshuachan.org/code.html) and joshuaccchan/bvar-toolkit-python (created 2026-10-11). CITING.md maps methods to papers; it has not been read. | 01:608r |
| `dieppe2016bear` | Dieppe, Legrand & van Roye (2016). The BEAR toolbox. ECB Working Paper 1934. | verified | ECB WP Series No 1934, July 2016 (PDF metadata). The bib carries the ECB PDF as url and the SSRN posting (DOI 10.2139/ssrn.2811020) as a note, so the two sources are not mixed. | 01:205 01:612r |
| `seabold2010statsmodels` | Seabold & Perktold (2010). statsmodels: econometric and statistical modeling with Python. | verified (completed) | Added *Proceedings of the 9th Python in Science Conference*, pp. 92–96, DOI. | 01:611r |
| `zanetta2024scoringrules` | scoringrules (Python; Zanetta & Allen?), version 0.11.0 used. | **corrected** | 'Zanetta & Allen?' resolved: Francesco Zanetta and Sam Allen. The README asks for @software 'scoringrules: a python library for probabilistic forecast evaluation' (2024). Version 0.11.0 (the version used, 02 §18) is on PyPI (2026-06-06). | 01:616r 02:1128 02:1717r |
| `bradbury2018jax` | Bradbury et al. (2018). JAX: composable transformations of Python+NumPy programs (software). | **corrected** | Author list per the JAX README citation block: 12 authors. 03 §16 item 22 lists 11 and omits Yash Katariya. Year 2018 (open-source release) as the README asks; version 0.12.0.1 (the version used) is on PyPI. | 01:617r 02:1679r 03:1586r 05:996r |
| `frostig2018compiling` | Frostig, Johnson & Leary (2018). Compiling machine learning programs via high-level tracing. SysML. | verified | Title block read from the PDF hosted at mlsys.org (SysML 2018). The Google Research page leaves Frostig out of the author list, but the PDF names him. | 03:1586r 03:1590r |
| `hoyer2017xarray` | Hoyer & Hamman (2017). xarray: N-D labeled arrays and datasets in Python. JORS 5(1). | verified (completed) | Added 5(1), article 10. | 01:618r 02:1699r |
| `kumar2019arviz` | Kumar, Carroll, Hartikainen & Martin (2019). ArviZ: a unified library for exploratory analysis of Bayesian models in Python. JOSS. | verified (completed) | Added 4(33), 1143. The official title has no colon: 'ArviZ a unified library …'. | 01:619r 04:1200r |
| `abrilpla2023pymc` | Abril-Pla et al. (2023). PyMC: a modern and comprehensive probabilistic programming framework in Python. | verified (completed) | Added PeerJ Computer Science 9:e1516. The official title has a comma: 'A modern, and comprehensive …'. | 01:620r |
| `phan2019composable` | Phan, Pradhan & Jankowiak (2019). Composable effects for flexible and accelerated probabilistic programming in NumPyro. | verified | arXiv:1912.11554 (NeurIPS 2019 workshop). | 01:621r |
| `harris2020array` | Harris et al. (2020). NumPy. Nature 585, 357-362. | verified | — | 05:995r |
| `virtanen2020scipy` | Virtanen et al. (2020). SciPy. Nature Methods 17, 261-272. | verified | The bib shortens the author list to the first 34 names plus 'SciPy 1.0 Contributors', in the order of the Crossref record. | 05:995r |
| `pymc2026extras` | pymc-extras BayesianVARMAX (landscape row). | verified | Exists. Version 0.15.1 (2026-09-16); `class BayesianVARMAX` is in pymc_extras/statespace/models/VARMAX.py. This does not check the features 01 §4.1 ascribes to it. | 01:610r |
| `pinder2026impulso` | impulso (Python; landscape row). | verified (completed) | Author Thomas Pinder, version 0.1.3 (2026-09-30), first release 2026-03-01, MIT (GitHub). The SPEC §1.4 descriptions are not checked. | 01:610r |
| `shaw2026srvar` | srvar-toolkit (Python; landscape row). | verified (completed) | Author Charles Shaw, version 0.4.0 (2026-09-27), first PyPI release 2026-07-18 (0.3.1). The SPEC §1.4 'first release Aug 2026' does not match PyPI; recheck the landscape row. | 01:610r |
| `fsf2026gplfaq` | Free Software Foundation. GPL FAQ, on the licence status of a program's output. | verified | Entries #GPLOutput and #WhatCaseIsOutputGPL. A web page, not a legal opinion (open question Q2 in 01). | 01:285 01:622r |
| `—` | Journal of Statistical Software, author guidelines. | excluded | Not a bibliography entry: it is a planning reference for 01 Q5. jstatsoft.org refused connections from this machine. A search index shows /authors and /about/submissions with the GPL-compatible-licence and replication-material requirements. Re-read the site before relying on it. | 01:565 01:623r |
