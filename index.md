# Selected projects in generative AI, machine learning, deep learing, and MLOps. 


---


## Fine-tuning LLMs with ORPO & QLoRA (Mistral-v0.3)
ORPO (Odds Ratio Preference Optimization) is a single-stage fine-tuning method to align LLMs with human preferences efficiently while preserving general performance and avoiding multi-stage training. This method trains directly on human preference pairs (chosen, rejected) without a reward model or reinforcement learning (RL) loop, reducing training complexity and resource usage. However, fine-tuning an LLM (e.g. full fine-tuning) for a particular task can still be computationally intensive as it involves updating all the LLM model parameters. Parameter-efficient fine-tuning (PEFT) updates only a small subset of parameters, allowing LLM fine-tuning with limited resources. Here, I have fine-tuned the [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3) foundation model with ORPO and QLoRA (a form of PEFT), by using NVIDIA L4 GPUs. In QLoRA, the pre-trained model weights are first quantized with 4-bit NormalFloat (NF4). The original model weights are frozen while trainable low-rank decomposition weight matrices are introduced and modified during the fine-tuning process, allowing for memory-efficient fine-tuning of the LLM without the need to retrain the entire model from scratch.  

[Check the model on Hugging Face hub!](https://huggingface.co/MuntasirHossain/Orpo-Mistral-7B-v0.3)


---
## Red-Teaming a Policy Assistant with Giskard

### Overview
This project demonstrates iterative red-teaming of a policy assistant designed to answer questions about a government-style digital services policy, while strictly avoiding legal advice, speculation, or guidance on bypassing safeguards. The focus is on safety evaluation, failure analysis, and mitigation, rather than model fine-tuning.

### Model Separation Strategy
The system intentionally uses **different models for generation and evaluation**:
* Query responses are generated using **gpt-4o-mini**
* Safety evaluation is performed using **gpt-4o** via Giskard detectors
This reflects common red-teaming practice: lighter models are sufficient for generation, while **stronger models provide more reliable safety judgments**. Separating generation and evaluation also avoids self-evaluation effects and keeps evaluation costs controlled.

### Initial Evaluation
The assistant was evaluated using **Giskard** across prompt-injection, misuse, and bias detectors. The scan identified multiple failures where the agent did not attempt to answer questions based on the provided policy document. These were not hallucinations or unsafe outputs, but overly conservative refusals.

<img src="images/giskard1.png?raw=true"/> Figure 1: Initial scan results from Giskard.

### Analysis
The root cause was **over-refusal**.
The safety layer correctly blocked requests involving legal advice, speculation, or bypassing safeguards, but also refused some benign questions that could have been partially answered using neutral policy language. This reduced policy grounding and triggered Giskard failures.

### Mitigation
The refusal strategy was refined to better distinguish between:
* questions requiring refusal, and
* questions that can be answered safely using policy text alone.
Refusals were standardized using fixed, auditable messages, while benign queries now trigger policy-based responses where possible. Safety guarantees were preserved.

### Outcome
A follow-up Giskard scan showed improved behavior:
* fewer false positives for “did not attempt to answer”
* stronger grounding in policy text
* no regression in prompt-injection or misuse resistance

<img src="images/giskard2.png?raw=true"/> Figure 2: Post mitigation scan results from Giskard.

This project demonstrates a complete red-teaming loop — evaluation, failure analysis, mitigation, and re-evaluation — and shows how safety behavior can be systematically improved without increasing risk or cost.

---

## Retrieval-Augmented Generation (RAG) with LLMs and Vector Databases
RAG is a technique that combines a retriever and a generative LLM to deliver accurate responses to queries. It involves retrieving relevant information from a large corpus and then generating contextually appropriate responses to queries. Here, I used the open-source Llama 3 and Mistral v2 models and LangChain with GPU acceleration to perform generative question-answering (QA) with RAG.

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/HuggingFace_Transformers-white?logo=huggingface)](#) [![](https://img.shields.io/badge/Jupyter-white?logo=Jupyter)](#) [![](https://img.shields.io/badge/PyTorch-white?logo=pytorch)](#)

[View example codes for introduction to RAG on GitHub](https://github.com/muntasirhsn/Introduction-to-Retrieval-Augmented-Generation)

**Try my app below that uses the Llama 3/Mistral v2 models and FAISS vector store for RAG on your PDF documents!**

<div>
  <iframe 
    src="https://muntasirhossain-rag-pdf-chatbot.hf.space" 
    width="770" 
    height="1000"
    style="border: 1px solid #d1d5db; border-radius: 8px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); transform:scale(0.95); transform-origin:0 0"
  ></iframe>
</div>


---

## MLOps with AWS: End-to-End ML Pipelines and Deployment
Develop an end-to-end machine learning (ML) workflow with automation for all the steps including data preprocessing, training models at scale with distributed computing (GPUs/CPUs), model evaluation, deploying in production, model monitoring and drift detection with Amazon SageMaker Pipeline - a purpose-built CI/CD service.


<img src="images/MLOps6_Muntasir Hossain.jpg?raw=true"/> Figure: ML orchestration reference architecture with AWS

<img src="images/Sageaker Pipeline5.png?raw=true"/> Figure: CI/CD pipeline with Amazon Sagemaker 

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![AWS](https://img.shields.io/badge/AWS-Cloud-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/)  [![Amazon Sagemaker](https://img.shields.io/badge/Sagemaker-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/sagemaker/) [![Amazon API Gateway](https://img.shields.io/badge/API_Gateway-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/api-gateway/) 

[View codes on GitHub](https://github.com/muntasirhsn/MLOps-with-AWS)




---


## LLM Alignment with Reinforcement Learning (RLHF & PPO) 
Fine-tuned a Google Flan-T5 language model for reduced toxicity using Reinforcement Learning with Human Feedback (RLHF). Implemented a PPO training loop with a learned reward model to guide generation toward lower-toxicity outputs, applying KL-divergence regularisation to maintain alignment with the base model. The model was adapted using parameter-efficient fine-tuning.

[Check the model on Hugging Face hub!](https://huggingface.co/MuntasirHossain/flan-t5-large-samsum-qlora-ppo)

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/PyTorch-white?logo=pytorch)](#) [![](https://img.shields.io/badge/HuggingFace_Transformers-white?logo=huggingface)](#) [![](https://img.shields.io/badge/Jupyter-white?logo=Jupyter)](#) 


---


## Neural Network-Based Time-Series Forecasting (CNN-LSTM)
This project implements a multi-step time-series forecasting model using a hybrid CNN-LSTM architecture. The 1D convolutional neural network (CNN) extracts local patterns (e.g., short-term fluctuations, trends) from the input sequence, while the LSTM network captures long-term temporal dependencies. Unlike recursive single-step prediction, the model performs direct multi-step forecasting (Seq2Seq), outputting am entire future sequence of values at once. Trained on historical energy data, the model forecasts weekly energy consumption over a consecutive 10-week horizon, achieving a Mean Absolute Percentage Error (MAPE) of 10% (equivalent to an overall accuracy of 90%). The results demonstrate robust performance for long-range forecasting, highlighting the effectiveness of combining CNNs for feature extraction and LSTMs for sequential modeling in energy demand prediction.

<iframe src="images/forecasting_2.html"
        width="650"
        height="350"
        frameborder="0"
        scrolling="no">
</iframe>
Figure: Actual and predicted energy usage over 10 weeks of time period.

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/TensorFlow-white?logo=TensorFlow)](#) [![](https://img.shields.io/badge/-Keras-white?logo=Keras&logoColor=black)](#) [![](https://img.shields.io/badge/Jupyter-white?logo=Jupyter)](#)

[View example codes on GitHub](https://github.com/muntasirhsn/CNN-LSTM-model-for-energy-usage-forecasting)

---


## Computer Vision: Deploying YOLOv8 at scale on AWS
YOLO (you only look once) is a state-of-the-art, real-time object detection and image segmentation model used in computer vision. The latest model YOLOv8 is  known for its runtime efficiency as well as detection accuracy. To fully utilise its potential, deploying the model at scale is crucial. Here, a YOLOv8 model was hosted on the Amazon SageMaker endpoint and inference was run for input images/videos for object detection.

<img src="images/highway1-detect3.gif?raw=true"/> Figure: Object detection with YOLOv8 model deployed to a real-time Amazon SageMaker endpoint

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/PyTorch-white?logo=pytorch)](#) [![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-white)](https://github.com/AlexeyAB/darknet) [![AWS](https://img.shields.io/badge/AWS-Cloud-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/) [![Amazon Sagemaker](https://img.shields.io/badge/Sagemaker-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/sagemaker/) 

[View project on GitHub](https://github.com/muntasirhsn/Deploying-YOLOv8-model-on-Amazon-SageMaker-endpoint)



---





---
<p style="font-size:11px">
