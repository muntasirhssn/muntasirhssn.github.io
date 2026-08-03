---
layout: default
---

# AI and Machine Learning Portfolio

A selection of applied AI, machine learning and MLOps projects demonstrating practical experience in agentic AI applications, cloud-based ML workflows, model deployment, automated reporting and predictive analytics. 

## Multi-Agent Healthcare Data Analyst

#### A deployed, privacy-aware agentic AI web application for first-pass healthcare data analysis and reporting.

This project is a multi-agent healthcare analytics web application designed to help healthcare teams convert their healthcare datasets into structured first-pass analysis reports. The system orchestrates specialised agents for analysis planning, code generation, controlled code execution and report writing, enabling automated data profiling, statistical summaries, visual exploration and narrative insight generation in a single workflow.

Designed as an assistive tool for healthcare analytics teams, the application helps accelerate early-stage data investigation by reducing repetitive analytical groundwork, improving reporting consistency and giving analysts, data scientists, clinicians or decision-makers a structured report to review, validate and extend. The project demonstrates practical experience in deployed agentic AI, multi-agent workflow design, Docker-based deployment and healthcare-oriented analytical decision support.

The project also explores privacy-aware design considerations and the practical use of open-weight language models such as GLM 5.2. This creates a pathway for organisations to adapt the solution for controlled infrastructure, where data control, secure deployment, reproducible analysis and reduced reliance on external closed-model APIs are important.

The application was tested using public healthcare datasets to validate the workflow and reporting capability.

**Try the deployed app:** 

<iframe 
    src="https://agentic-data-analyst-production-cb1d.up.railway.app/" 
    width="650" 
    height="1200px" 
    frameborder="0"
    style="border: 1px solid #ddd; border-radius: 8px;"
></iframe>

![](https://img.shields.io/badge/Python-white?logo=Python) ![](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)


---


## End-to-End ML Pipelines and Deployment at Scale

This project demonstrates an end-to-end machine learning workflow on AWS, covering data ingestion, preprocessing, model training, evaluation, deployment and monitoring concepts. The workflow uses Amazon SageMaker Pipelines for ML orchestration and integrates CI/CD practices to support repeatable, production-oriented model delivery.

The project highlights practical MLOps concepts including automated pipeline stages, scalable training, model evaluation, deployment readiness, monitoring and drift detection. It is designed to show how machine learning workflows can be structured for reproducibility, reliability and operational scalability in cloud environments.


<img src="images/MLOps6_Muntasir Hossain.jpg?raw=true"/> 
Figure: ML orchestration reference architecture with AWS

<img src="images/Sageaker Pipeline5.png?raw=true"/> 

Figure: CI/CD pipeline with Amazon Sagemaker 

[View codes on GitHub](https://github.com/muntasirhsn/MLOps-with-AWS)

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![AWS](https://img.shields.io/badge/AWS-Cloud-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/)  [![Amazon Sagemaker](https://img.shields.io/badge/Sagemaker-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/sagemaker/) [![Amazon API Gateway](https://img.shields.io/badge/API_Gateway-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/api-gateway/) 


---


## Neural Network-Based Time-Series Forecasting

This project implements a multi-step time-series forecasting model using a hybrid CNN-LSTM architecture. The 1D convolutional neural network extracts local temporal patterns such as short-term fluctuations and trends, while the LSTM network captures longer-term sequence dependencies.

Unlike recursive single-step forecasting, the model performs direct multi-step prediction using a sequence-to-sequence approach, outputting an entire future sequence at once. Trained on historical energy consumption data, the model forecasts weekly energy demand over a consecutive 10-week horizon and achieved a Mean Absolute Percentage Error of 10%, equivalent to approximately 90% forecast accuracy.

The project demonstrates practical experience in neural-network-based forecasting, sequence modelling, model evaluation and applied decision-support analytics for demand planning.

<iframe src="images/forecasting_2.html"
        width="850"
        height="350"
        frameborder="0"
        scrolling="no">
</iframe>
Figure: Actual and predicted energy usage over 10 weeks of time period.

[View example codes on GitHub](https://github.com/muntasirhsn/CNN-LSTM-model-for-energy-usage-forecasting)

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/TensorFlow-white?logo=TensorFlow)](#) [![](https://img.shields.io/badge/-Keras-white?logo=Keras&logoColor=black)](#) [![](https://img.shields.io/badge/Jupyter-white?logo=Jupyter)](#)




---
<p style="font-size:11px">
