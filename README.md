# 192.151 Introduction to Deep Learning  
**2025S**  
## Project 4: Sentiment Analysis of Text

---

### 1. Background and Motivation

Sentiment analysis, the task of determining the emotional tone (positive, negative, neutral) behind a piece of text, is a cornerstone of Natural Language Processing (NLP). It has wide applications, from gauging public opinion on social media to understanding customer feedback. 

Traditional methods often rely on bag-of-words or TF-IDF features. Deep learning models, particularly Recurrent Neural Networks (RNNs like LSTMs or GRUs) and more recently Transformers (like BERT or DistilBERT), have shown superior performance by learning rich contextual representations of text. This project provides an opportunity to explore these advanced techniques.

---

### 2. Problem Description

This project involves building and evaluating deep learning models (RNNs or Transformers) for sentiment classification of text, such as movie reviews or product feedback. This involves:

#### a) Dataset Preparation and Preprocessing:
- Obtain a binary or multi-class sentiment classification dataset. (e.g., IMDb, SST-2, or SemEval).
- Preprocessing text data: tokenization, handling punctuation, creating vocabulary, padding sequences.
- Implementing word embeddings (e.g., pre-trained like Word2Vec/GloVe, or learned from scratch).

#### b) Sentiment Analysis Model Implementation:
- Implement an RNN-based model (e.g., LSTM or GRU layers).
- *(Optional but encouraged)* implement a Transformer-based model (e.g., fine-tuning a pre-trained DistilBERT or building a simpler Transformer encoder).
- Handle input formatting, embedding layers, and output layers for classification.

#### c) Training and Evaluation:
- Train the implemented models on the chosen dataset.
- Experiment with hyperparameters (e.g., embedding size, RNN hidden units, attention mechanism details, learning rate, dropout).
- Evaluate model performance comprehensively (e.g., accuracy, precision, recall, F1-score).
- Compare the performance of the different models implemented.

---

### 3. Expected Deliverables

Students working in groups (ideally 3 members) are expected to:

- Produce a project report,  
- Deliver a group presentation, and  
- Submit well-commented source code.

---

### 4. Suggested Resources and References

1. **Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014)**.  
   *Generative Adversarial Nets*. In Advances in Neural Information Processing Systems (NeurIPS), 27.  
   [https://arxiv.org/pdf/1406.2661](https://arxiv.org/pdf/1406.2661)

2. **Karras, T., Laine, S., & Aila, T. (2019)**.  
   *A Style-Based Generator Architecture for Generative Adversarial Networks*. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 4401–4410).  
   [https://arxiv.org/pdf/1812.04948](https://arxiv.org/pdf/1812.04948)

3. **Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022)**.  
   *High-Resolution Image Synthesis with Latent Diffusion Models*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 10684–10695).  
   [https://arxiv.org/pdf/2112.10752](https://arxiv.org/pdf/2112.10752)

