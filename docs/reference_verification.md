# Reference verification for Methodology and Results

Verified 2026-09-10 against the original publisher/repository/dataset pages. This file records the exact supported use; it is not a literature review.

| Ref. | Original source reread | Supported use in this project | Boundary |
|---|---|---|---|
| [1], [2] | Reyna et al. paper metadata and PhysioNet Challenge v1.0.0 page | 40 predictor columns including `Hct`; 40,336 public A/B subjects; persistent `SepsisLabel` shifted six hours | Does not make the present analysis an independent external validation. |
| [3] | Official evaluation repository at pinned commit `467c49b...` and vendored source hash `26b8b2...` | Piecewise patient utility constants and official normalization | Does not justify a custom scorer or use reconstructed onset in Challenge Utility. |
| [4] | Richman and Moorman original article/PubMed record | SampEn as negative log of conditional template-match probability excluding self-matches | Short, missing-value windows still require explicit implementation rules. |
| [5] | Varma and Simon full text | Inner CV for tuning and outer CV for error estimation; all tuning repeated within the loop | Supports the architecture, not claims of external validity. |
| [6] | Chen and Guestrin paper/author preprint | Additive boosted-tree model, regularized objective and sparsity-aware learning | Implementation parameters come from code/XGBoost 1.7.6, not from the paper. |
| [7] | Van Calster et al. full text | Calibration-in-the-large/intercept target 0, slope target 1 and need for calibration assessment | Does not imply that Platt recalibration improves every metric. |
| [8] | Vickers and Elkin full text | Net benefit `TP/n - FP/n * pt/(1-pt)` and comparison with treat-all/treat-none | DCA is conditional on the stated clinical action and threshold range. |
| [9] | Benjamini and Hochberg publisher metadata | False-discovery-rate adjustment for the declared three-metric family | The implementation and achieved Monte Carlo resolution remain code/run properties. |
| [10] | Manis et al. full text | Missing observations can change SampEn computation and must be handled explicitly | Does not validate imputing missing observations as physiological samples; this pipeline uses observed values only for SampEn. |
| [11] | Collins et al. publisher metadata | Reporting transparency for prediction-model studies using regression/ML | Reporting guidance is not a risk-of-bias PASS. |
| [12] | Moons et al. publisher metadata | PROBAST+AI is a structured appraisal of quality, risk of bias and applicability | A machine-generated status cannot replace human signalling responses/rationale. |
| [13] | scikit-learn 1.2 API definitions used by the run | Exact software estimands for `roc_auc_score`, `average_precision_score`, `precision_recall_curve`, `auc` and `brier_score_loss` | XGBoost `aucpr` is not used or relabelled as sklearn Average Precision. |

## IEEE references

[1] M. A. Reyna, C. S. Josef, R. Jeter, S. P. Shashikumar, M. B. Westover, S. Nemati, G. D. Clifford, and A. Sharma, “Early prediction of sepsis from clinical data: The PhysioNet/Computing in Cardiology Challenge 2019,” *Crit. Care Med.*, vol. 48, no. 2, pp. 210–217, Feb. 2020, doi: 10.1097/CCM.0000000000004145.

[2] M. Reyna *et al.*, “Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019,” PhysioNet, ver. 1.0.0, 2019, doi: 10.13026/V64V-D857. [Online]. Available: https://physionet.org/content/challenge-2019/1.0.0/

[3] PhysioNet Challenges, “Evaluation code for the PhysioNet/CinC Challenge 2019,” GitHub, commit `467c49b514542be7a4a0bafe40fa2c3b064dda2e`, 2019. [Online]. Available: https://github.com/physionetchallenges/evaluation-2019

[4] J. S. Richman and J. R. Moorman, “Physiological time-series analysis using approximate entropy and sample entropy,” *Am. J. Physiol.-Heart Circ. Physiol.*, vol. 278, no. 6, pp. H2039–H2049, Jun. 2000, doi: 10.1152/ajpheart.2000.278.6.H2039.

[5] S. Varma and R. Simon, “Bias in error estimation when using cross-validation for model selection,” *BMC Bioinformatics*, vol. 7, Art. no. 91, 2006, doi: 10.1186/1471-2105-7-91.

[6] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, San Francisco, CA, USA, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.

[7] B. Van Calster *et al.*, “Calibration: The Achilles heel of predictive analytics,” *BMC Med.*, vol. 17, Art. no. 230, 2019, doi: 10.1186/s12916-019-1466-7.

[8] A. J. Vickers and E. B. Elkin, “Decision curve analysis: A novel method for evaluating prediction models,” *Med. Decis. Making*, vol. 26, no. 6, pp. 565–574, 2006, doi: 10.1177/0272989X06295361.

[9] Y. Benjamini and Y. Hochberg, “Controlling the false discovery rate: A practical and powerful approach to multiple testing,” *J. Roy. Stat. Soc. B*, vol. 57, no. 1, pp. 289–300, 1995, doi: 10.1111/j.2517-6161.1995.tb02031.x.

[10] G. Manis, D. Platakis, and R. Sassi, “Sample entropy computation on signals with missing values,” *Entropy*, vol. 26, no. 8, Art. no. 704, Aug. 2024, doi: 10.3390/e26080704.

[11] G. S. Collins *et al.*, “TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods,” *BMJ*, vol. 385, Art. no. e078378, 2024, doi: 10.1136/bmj-2023-078378.

[12] K. G. M. Moons *et al.*, “PROBAST+AI: An updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods,” *BMJ*, vol. 388, Art. no. e082505, 2025, doi: 10.1136/bmj-2024-082505.

[13] scikit-learn Developers, “Metrics and scoring: Quantifying the quality of predictions,” *scikit-learn 1.2 Documentation*. [Online]. Available: https://scikit-learn.org/1.2/modules/model_evaluation.html. Accessed: Sep. 10, 2026.
