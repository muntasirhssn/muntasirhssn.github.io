# Selected projects in machine learning, deep learing, generative AI, data science, and computer vision. 


---

## Retrieval-Augmented Generation (RAG) with LLMs, embeddings, vector databases, and LangChain
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

## Deep Learning: Multi-step energy usage forecasting with CNN-LSTM architecture
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

## MLOps with AWS: Train and deploy ML models at scale with automated pipelines
Develop an end-to-end machine learning (ML) workflow with automation for all the steps including data preprocessing, training models at scale with distributed computing (GPUs/CPUs), model evaluation, deploying in production, model monitoring and drift detection with Amazon SageMaker Pipeline - a purpose-built CI/CD service.


<img src="images/MLOps6_Muntasir Hossain.jpg?raw=true"/> Figure: ML orchestration reference architecture with AWS

<img src="images/Sageaker Pipeline5.png?raw=true"/> Figure: CI/CD pipeline with Amazon Sagemaker 

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![AWS](https://img.shields.io/badge/AWS-Cloud-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/)  [![Amazon Sagemaker](https://img.shields.io/badge/Sagemaker-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/sagemaker/) [![Amazon API Gateway](https://img.shields.io/badge/API_Gateway-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/api-gateway/) 

[View codes on GitHub](https://github.com/muntasirhsn/MLOps-with-AWS)


---


## Parameter-efficient fine-tuning of LLMs with Quantized Low-Rank Adaptation (QLoRA)
Fine-tuning an LLM (e.g. full fine-tuning) for a particular task can be computationally intensive as it involves updating all the LLM model parameters. Parameter-efficient fine-tuning (PEFT) updates only a small subset of parameters, allowing LLM fine-tuning with limited resources. Here, I have fine-tuned the [Llama-3-8B](meta-llama/Meta-Llama-3-8B) foundation model with QLoRA, a form of PEFT, by using NVIDIA L4 GPUs. In QLoRA, the pre-trained model weights are first quantized with 4-bit NormalFloat (NF4). The original model weights are frozen while trainable low-rank decomposition weight matrices are introduced and modified during the fine-tuning process, allowing for memory-efficient fine-tuning of the LLM without the need to retrain the entire model from scratch.  

[Check the model on Hugging Face hub!](https://huggingface.co/MuntasirHossain/Meta-Llama-3-8B-OpenOrca)



---


## Reinforcement Learning with Human Feedback (RLHF) 
Reinforcement Learning with Human Feedback (RLHF) is a cutting-edge approach used to fine-tune Large Language Models (LLMs) to generate outputs that align more closely with human preferences and expectations. Here, I utilised RLHF to further fine-tune a [Google Flan-T5 Large](https://huggingface.co/MuntasirHossain/flan-t5-large-samsum-qlora) and generate less toxic content. The model was previously fine-tuned for generative summarisation with PEFT. A [binary classifier model](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target) from Meta AI was used as the reward model to score and reward the LLM output based on the toxicity level. The LLM was fine-tuned with Proximal Policy Optimization (PPO) using those reward values. The iterative process for maximising cumulative rewards and fine-tuning with PPO enables detoxified LLM outputs. To ensure that the model does not deviate from generating content that is too far from the original LLM, KL-divergence was employed during the iterative training process.

[Check the model on Hugging Face hub!](https://huggingface.co/MuntasirHossain/flan-t5-large-samsum-qlora-ppo)

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/PyTorch-white?logo=pytorch)](#) [![](https://img.shields.io/badge/HuggingFace_Transformers-white?logo=huggingface)](#) [![](https://img.shields.io/badge/Jupyter-white?logo=Jupyter)](#) 



---

## Analysing the global CO₂ emission: How did countries, regions and economic zones evolve over time?

The World Bank provides data for greenhouse gas emissions in million metric tons of CO₂ equivalent (Mt CO₂e) based on the AR5 global warming potential (GWP). It provides key insights into environmental impact at both national, regional and economic levels over the past six decades.
* Many countries (e.g. China, India) and regions show a clear upward trend in carbon dioxide emissions from 1960 to 2024. For instance, although the population of China increased from 0.82 billion in 1970 to only 1.41 billion in 2023, emission drastically increased from 909 Mt CO₂e to over 13000 Mt CO₂e, reflecting rapid industrialization. While the population of China and India in 2023 were nearly the same, the CO₂ emission from China was 4.5 fold to that of India. 
* There is significant variation between countries. Highly industrialized or resource-rich nations (e.g., Saudi Arabia, United Arab Emirates) emit far more CO₂ than smaller or less industrialized countries (e.g., Aruba, Burundi).
* The data suggests a strong link between economic development and emissions growth. Countries experiencing rapid economic expansion (e.g., Vietnam, United Arab Emirates) show marked increases in emissions, while some developed countries (e.g., Germany, Austria, Belgium) have stabilized or slightly reduced their emissions in recent years, likely due to policy interventions or shifts to cleaner energy.
* While ‘High Income’ regions dominated the CO2 emissions pre-2020, the ‘Middle Income’ and ‘Upper Middle Income’ regions rapidly increased CO2 emissions after 2000, exceeding the emissions from ‘High Income’ regions. 

### Global CO₂ emissions
<iframe src="images/co2_emissions_world_animation_start_on_2023_fixed.html"
        width="700"
        height="650"
        frameborder="0"
        scrolling="no">
</iframe>
Figure: Interactive visualization of global CO₂ emissions by country and year



### Time sereis CO₂ emissions
<iframe src="images/co2_emissions_timeseries_trend.html"
        width="650"
        height="550"
        frameborder="0"
        scrolling="no">
</iframe>
Figure: Time sereis CO₂ emissions for selected countries



### Population Growth
<iframe src="images/Population_timeseries.html"
        width="650"
        height="550"
        frameborder="0"
        scrolling="no">
</iframe>
Figure: Population growth for selected countries



### CO₂ emissions by income groups
<iframe src="images/co2_emissions_bar_income_zone_cleaned.html"
        width="650"
        height="550"
        frameborder="0"
        scrolling="no">
</iframe>
Figure: Interactive visualization of CO₂ emissions for different income zones from 1970 to 2023



### CO₂ emissions by geographic regions
<iframe src="images/co2_emissions_pie_fixed_colors_position.html"
        width="650"
        height="550"
        frameborder="0"
        scrolling="no">
</iframe>
Figure: Interactive visualization of CO₂ emissions for different geographic regions from 1970 to 2023



---


## Computer Vision: Deploying YOLOv8 model at scale with Amazon Sagemaker Endpoints
YOLO (you only look once) is a state-of-the-art, real-time object detection and image segmentation model used in computer vision. The latest model YOLOv8 is  known for its runtime efficiency as well as detection accuracy. To fully utilise its potential, deploying the model at scale is crucial. Here, a YOLOv8 model was hosted on the Amazon SageMaker endpoint and inference was run for input images/videos for object detection.

<img src="images/highway1-detect3.gif?raw=true"/> Figure: Object detection with YOLOv8 model deployed to a real-time Amazon SageMaker endpoint

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/PyTorch-white?logo=pytorch)](#) [![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-white)](https://github.com/AlexeyAB/darknet) [![AWS](https://img.shields.io/badge/AWS-Cloud-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/) [![Amazon Sagemaker](https://img.shields.io/badge/Sagemaker-white?logo=amazon-aws&logoColor=orange)](https://aws.amazon.com/sagemaker/) 

[View project on GitHub](https://github.com/muntasirhsn/Deploying-YOLOv8-model-on-Amazon-SageMaker-endpoint)



---





---
<p style="font-size:11px">
