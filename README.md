AI-powered healthcare payment integrity system that can detect unusual claim behavior and give it a risk score so a human can review it.
For example:
The system gets a healthcare claim with the following:
Claim ID: 121212
Provider ID: 123
Patient ID: 321
Procedure: MRI
Diagnosis: Knee injury
Claim Amount: $88,500
Date: 2026-05-10

System classifies it to Normal or High-Risk claim with explanation: " claim is x10 higher than provider average or provider submittied unusually high monthly volume" 
Then a human reviews it.

System Architecture:
              Healthcare Claims Dataset
                        |
                        v
              Data Cleaning Pipeline
                        |
                        v
              Feature Engineering
                        |
                        v
              ML Anomaly Detector
             (Isolation Forest)
                        |
                        v
              Risk Scoring Engine
                        |
                        v
              Explanation Generator
              (Rules / LLM)
                        |
                        v
              Streamlit Dashboard


Dataset source: https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis
