ChatRec_Model - Conversational Response Prediction System
==========================================================

This project implements an offline GPT-2 Transformer model to predict User A's next reply based on conversational context.

----------------------------------------------------------
Directory Structure
----------------------------------------------------------
ChatRec_Model/
│── ChatRec_Model.ipynb      : Jupyter notebook containing code execution
│── Report.pdf               : Technical report summarizing model design & evaluation
│── Model.joblib             : Serialized fine-tuned GPT-2 model and tokenizer
│── ReadMe.txt               : Documentation file (this one)

----------------------------------------------------------
Steps to Run
----------------------------------------------------------
1. Install dependencies:
   pip install torch transformers datasets nltk rouge-score sacrebleu pandas scikit-learn joblib tqdm

2. Place your conversation dataset (conversationfile.xlsx or .csv) in the working directory.

3. Run ChatRec_Model.ipynb to preprocess, train, and evaluate the model.

4. After training:
   - The model will be saved as ./final_model/ (Hugging Face format)
   - It will also be serialized as ChatRec_Model/Model.joblib

----------------------------------------------------------
Evaluation Summary
----------------------------------------------------------
BLEU Score : 0.0102
ROUGE-1    : 0.1500
ROUGE-2    : 0.0000
ROUGE-L    : 0.1000
Perplexity : 53.9670

----------------------------------------------------------
Model Justification
----------------------------------------------------------
Model  : GPT-2 (autoregressive Transformer)
Reason : Efficient for small datasets, capable of context-aware text generation.
Deployment : Joblib serialization allows fast loading in production (Flask/FastAPI).

----------------------------------------------------------
Author's Note
----------------------------------------------------------
This model serves as a prototype. For production, train with more data and consider
larger models (GPT-2 Medium, T5, or Longformer) for better context understanding.
