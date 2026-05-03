# Streamlit Demo Examples

Run this after regenerating the processed dataset to refresh clean examples:

```powershell
python scripts/07_create_demo_examples.py
```

In the current dataset source, some older processed files may store the abstract in `instruction` and the question in `input`. The helper script normalizes this into clear Question and Context sections.

## Example 1

**Question**

Answer the question based on the following context: Adverse drug events represent the most common cause of preventable nonsurgical adverse events in medicine but may remain undetected. Our objective is to determine the proportion of drug-related visits emergency physicians attribute to medication-related problems. This prospective observational study enrolled adults presenting to a tertiary care emergency department (ED) during 12 weeks. Drug-related visits were defined as ED visits caused by adverse drug events. The definition of adverse drug event was varied to examine both narrow and broad adverse drug event classification systems. Clinical pharmacists evaluated all patients for drug-related visits, using standardized assessment algorithms, and then followed patients until hospital discharge. Interrater agreement for the clinical pharmacist diagnosis of drug-related visit was assessed. Emergency physicians, blinded to the clinical pharmacist opinion, were interviewed at the end of each shift to determine whether they attributed the visit to a medication-related problem. An independent committee reviewed and adjudicated all cases in which the emergency physicians' and clinical pharmacists' assessments were discordant, or either the emergency physician or clinical pharmacist was uncertain. The primary outcome was the proportion of drug-related visits attributed to a medication-related problem by emergency physicians. Nine hundred forty-four patients were enrolled, of whom 44 patients received a diagnosis of the narrowest definition of an adverse drug event, an adverse drug reaction (4.7%; 95% confidence interval [CI] 3.5% to 6.2%). Twenty-seven of these were categorized as medication-related by emergency physicians (61.4%; 95% CI 46.5% to 74.3%), 10 were categorized as uncertain (22.7%; 95% CI 12.9% to 37.1%), and 7 categorized as a non-medication-related problem (15.9%; 95% CI 8.0% to 29.5%). Seventy-eight patients (8.3%; 95% CI 6.7% to 10.2%) received a diagnosis of an adverse drug event caused by an adverse drug reaction, a drug interaction, drug withdrawal, a medication error, or noncompliance. Emergency physicians attributed 49 of these to a medication-related problem (62.8%; 95% CI 51.7% to 72.7%), were uncertain about 15 (19.2%; 95% CI 12.0% to 29.4%), and attributed 14 to non-medication-related problems (17.9%; 95% CI 11.0% to 27.9%). Twenty-five of 29 (86.2%; 95% CI 69.3% to 94.4%) adverse drug events not considered medication related by emergency physicians were rated at least moderate in severity.

**Context**

Question: Do emergency physicians attribute drug-related emergency department visits to medication-related problems?

**Reference Answer**

A significant proportion of drug-related visits are not deemed medication related by emergency physicians. Drug-related visits not attributed to medication-related problems by emergency physicians may be missed in ongoing outpatient adverse drug event surveillance programs intended to develop strategies to enhance drug safety. Further research is needed to determine what the effect may be of not attributing adverse drug events to medication-related problems.

## Example 2

**Question**

Answer the question based on the following context: Our objective was to determine whether measurement of placenta growth factor (PLGF), inhibin A, or soluble fms-like tyrosine kinase-1 (sFlt-1) at 2 times during pregnancy would usefully predict subsequent preeclampsia (PE) in women at high risk. We analyzed serum obtained at enrollment (12(0/7) to 19(6/7) weeks) and follow-up (24-28 weeks) from 704 patients with previous PE and/or chronic hypertension (CHTN) enrolled in a randomized trial for the prevention of PE. Logistic regression analysis assessed the association of log-transformed markers with subsequent PE; receiver operating characteristic analysis assessed predictive value. One hundred four developed preeclampsia: 27 at 37 weeks or longer and 77 at less than 37 weeks (9 at less than 27 weeks). None of the markers was associated with PE at 37 weeks or longer. Significant associations were observed between PE at less than 37 weeks and reduced PLGF levels at baseline (P = .022) and follow-up (P<.0001) and elevated inhibin A (P<.0001) and sFlt-1 (P = .0002) levels at follow-up; at 75% specificity, sensitivities ranged from 38% to 52%. Using changes in markers from baseline to follow-up, sensitivities were 52-55%. Associations were observed between baseline markers and PE less than 27 weeks (P<or = .0004 for all); sensitivities were 67-89%, but positive predictive values (PPVs) were only 3.4-4.5%.

**Context**

Question: Serum inhibin A and angiogenic factor levels in pregnancies with previous preeclampsia and/or chronic hypertension: are they useful markers for prediction of subsequent preeclampsia?

**Reference Answer**

Inhibin A and circulating angiogenic factors levels obtained at 12(0/7) to 19(6/7) weeks have significant associations with onset of PE at less than 27 weeks, as do levels obtained at 24-28 weeks with onset of PE at less than 37 weeks. However, because the corresponding sensitivities and/or PPVs were low, these markers might not be clinically useful to predict PE in women with previous PE and/or CHTN.

## Example 3

**Question**

Answer the question based on the following context: Fewer emergency department (ED) visits may be a potential indicator of quality of care during the end of life. Receipt of palliative care, such as that offered by the adult Palliative Care Service (PCS) in Halifax, Nova Scotia, is associated with reduced ED visits. In June 2004, an integrated service model was introduced into the Halifax PCS with the objective of improving outcomes and enhancing care provider coordination and communication. The purpose of this study was to explore temporal trends in ED visits among PCS patients before and after integrated service model implementation. PCS and ED visit data were utilized in this secondary data analysis. Subjects included all adult patients enrolled in the Halifax PCS between January 1, 1999 and December 31, 2005, who had died during this period (N = 3221). Temporal trends in ED utilization were evaluated dichotomously as preintegration or postintegration of the new service model and across 6-month time blocks. Adjustments for patient characteristics were performed using multivariate logistic regression. Fewer patients (29%) made at least one ED visit postintegration compared to the preintegration time period (36%, p<0.001). Following adjustments, PCS patients enrolled postintegration were 20% less likely to have made at least one ED visit than those enrolled preintegration (adjusted OR 0.8; 95% confidence interval 0.6-1.0).

**Context**

Question: Can the introduction of an integrated service model to an existing comprehensive palliative care service impact emergency department visits among enrolled patients?

**Reference Answer**

There is some evidence to suggest the introduction of the integrated service model has resulted in a decline in ED visits among PCS patients. Further research is needed to evaluate whether the observed reduction persists.

## Example 4

**Question**

Answer the question based on the following context: The role of eosinopenia as a marker of sepsis has recently been evaluated. The aim of our study was to test the value of eosinopenia as a diagnostic marker of sepsis in comparison to procalcitonin and C-reactive protein levels. A prospective study of critically ill adult patients admitted to the medical intensive care unit at an urban hospital. Procalcitonin, C-reactive protein (CRP) levels and eosinophil counts were measured on admission. Patients were classified as non-infected or infected by the medical residents, fellows, and attendings. A total of 68 patients were enrolled into the study. At a cut-off value of 70 mg/L, the CRP level yielded a sensitivity of 94%, a specificity of 84%, a positive predicted value (PPV) of 83% and a negative predicted value (NPV) of 94%. At a cutoff value of 1.5 μg/L, the sensitivity of the procalcitonin test was 84%, specificity of 92%, PPV 90%, and NPV of 87%. The eosinophil cell count (cutoff of 50 cells/mm(3)) produced a sensitivity of 81%, specificity of 65%, a PPV of 66%, and a NPV of 80%. The comparison of the eosinophil cell count (<50 cells/mm(3)) and procalcitonin levels among the non-infected and infected groups showed a significant statistical difference (Fisher exact test, P = .0239). There was no statistical difference observed when comparisons were made between CRP levels and eosinophil count (Fisher exact test, P = .12). There was also a lack of significant statistical difference when CRP levels were compared to procalcitonin levels (Fisher exact test, P = .49).

**Context**

Question: Eosinopenia: Is it a good marker of sepsis in comparison to procalcitonin and C-reactive protein levels for patients admitted to a critical care unit in an urban hospital?

**Reference Answer**

Eosinopenia is a very sensitive yet not specific serological marker of sepsis in the intensive care unit and can be utilized to guide physicians in the diagnosis of sepsis.

## Example 5

**Question**

Answer the question based on the following context: Several randomized controlled trials and observational studies have compared outcomes for coronary artery bypass graft (CABG) surgery and drug-eluting stents (DES), but these studies have not thoroughly investigated the relative difference in outcomes by sex. We aimed to compare 3-year outcomes (mortality, mortality/myocardial infarction/stroke, and repeat revascularization) for CABG surgery and percutaneous coronary interventions with DES by sex. A total of 4,532 women (2,266 pairs of CABG and DES patients) and 11,768 men (5,884 pairs) were propensity matched separately using multiple patient risk factors and were compared with respect to 3-year outcomes. Both women and men receiving DES had significantly higher mortality rates (adjusted hazard ratio, 1.28; 95% confidence interval, 1.06 to 1.54 and adjusted hazard ratio, 1.22; 95% confidence interval, 1.06 to 1.41, respectively) and myocardial infarction/mortality/stroke rates (adjusted hazard ratio, 1.40; 95% confidence interval, 1.19 to 1.64 and adjusted hazard ratio, 1.36; 95% confidence interval, 1.20 to 1.54, respectively) with DES. The advantage for CABG surgery was also present for several preselected patient subgroups. Men had consistently lower adverse outcome rates than women for both procedures. For example, the mortality rates for CABG and DES for men were 8.0% and 9.1%, compared with respective rates of 11.8% and 13.7% for women.

**Context**

Question: Comparison of 3-Year Outcomes for Coronary Artery Bypass Graft Surgery and Drug-Eluting Stents: Does Sex Matter?

**Reference Answer**

For women, the advantage of CABG surgery over DES is very similar to what was found for men, and this advantage persisted for patients with and without high-risk characteristics.
