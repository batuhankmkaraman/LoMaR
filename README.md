This repository is released under an Academic Research-Only License (no commercial use, no redistribution, no patent license).

# LoMaR
## Code for the paper titled "Longitudinal Mammogram Risk Prediction". 

Python implementation of LoMaR. See [abstract](#abstract) and [paper](#citation) for more details.

# Citation 

If you use the open source code, please cite:  

-  Deep-learning-based Optimization of the Under-sampling Pattern in MRI  
C. Bahadir‡, A.Q. Wang‡, A.V. Dalca, M.R. Sabuncu.  
IEEE TCP: Transactions on Computational Imaging. 6. pp. 1139-1152. 2020.  [arXiv:1907.11374](https://arxiv.org/abs/1907.11374).

- Learning-based Optimization of the Under-sampling Pattern in MRI.  
Cagla D. Bahadir, Adrian V. Dalca, and Mert R. Sabuncu.  
IPMI: Information Processing in Medical Imaging. 2019. [arXiv:1901.01960](https://arxiv.org/abs/1901.01960).

# Abstract
Breast cancer is one of the leading causes of mortality among
women worldwide. Early detection and risk assessment play a crucial role
in improving survival rates. Therefore, annual or biennial mammograms
are often recommended for screening in high-risk groups. Mammograms
are typically interpreted by expert radiologists based on the Breast Imag-
ing Reporting and Data System (BI-RADS), which provides a uniform
way to describe findings and categorizes them to indicate the level of
concern for breast cancer. Recently, machine learning (ML) and com-
putational approaches have been developed to automate and improve
the interpretation of mammograms. However, both BI-RADS and the
ML-based methods focus on the analysis of data from the present and
sometimes the most recent prior visit. While it is clear that temporal
changes in image features of the longitudinal scans should carry value for
quantifying breast cancer risk, no prior work has conducted a systematic
study of this. In this paper, we extend a state-of-the-art ML model [20]
to ingest an arbitrary number of longitudinal mammograms and pre-
dict future breast cancer risk. On a large-scale dataset, we demonstrate
that our model, LoMaR, achieves state-of-the-art performance when pre-
sented with only the present mammogram. Furthermore, we use LoMaR
to characterize the predictive value of prior visits. Our results show that
longer histories (e.g., up to four prior annual mammograms) can signifi-
cantly boost the accuracy of predicting future breast cancer risk, partic-
ularly beyond the short-term.

# Trained Model Weights
Trained model weights for LoMaR are available upon request due to large file sizes. Please contact Batuhan Karaman (batuhankmkaraman@gmail.com) for the weight files.

