# Project report outline - FraudLens AI
1. **Title page, certificate, acknowledgements, abstract** (problem, data, models, best result, main limitation).
2. **Introduction** - digital payments, cost of fraud, why ML; problem statement.
3. **Objectives** - the seven objectives in README.md.
4. **Related work** - PaySim (Lopez-Rojas et al.), IEEE-CIS (Vesta), ULB credit-card dataset, class imbalance and PR-AUC for rare events, temporal evaluation (add 5-8 references).
5. **Dataset** - how the Mendeley package was built (balanced, time-matched samples from three sources), schema, per-source sizes, **why it is balanced and what that does to metrics**, why PaySim is simulated and not UPI.
6. **Methodology** - integrity checks, time encoding, features, time vs random splits, CV-based selection, threshold choice, base-rate projection.
7. **System design** - architecture (browser -> FastAPI -> pipeline/model/data), API table, prediction data flow.
8. **Implementation** - stack, key modules, screenshots of each page.
9. **EDA findings** - notebook plots 2.1-2.6 with 2-3 sentences each.
10. **Results** - model table under both protocols, confusion matrix, curves, per-source AUC, feature importance, real-rate projection (precision about 1 % / 5 % / 0.2 % at real prevalence).
11. **Testing** - the API and browser checks and what they verify.
12. **Limitations and ethics** - small balanced sample, mixed sources, weak card-source features, source as input, approximate prevalence, privacy and fairness, cost of false positives.
13. **Future work** - full-size datasets with real prevalence, per-source models, richer features (balances, device, velocity), SHAP, cost-sensitive thresholds, monitoring for drift.
14. **Conclusion, references, appendix.**
