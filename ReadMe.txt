ChatRec_Model — Conversational Response Prediction System
=========================================================

Overview
--------
ChatRec_Model is an offline GPT-2–based system designed to predict a user’s next conversational reply from previous dialogue context.  
It demonstrates how a transformer-based model can be fine-tuned and deployed for conversational response generation in lightweight environments.

----------------------------------------------------------
Directory Structure
----------------------------------------------------------
ChatRec_Model/
│── ChatRec_Model.ipynb      — Jupyter notebook with preprocessing, training, and evaluation steps  
│── final_model/             — Saved Hugging Face model files after training  
│── Model.joblib             — Serialized version of the fine-tuned GPT-2 model and tokenizer  
│── Report.pdf               — Technical report summarizing architecture, results, and analysis  
│── ReadMe.txt               — This documentation file  

----------------------------------------------------------
Steps to Run
----------------------------------------------------------
1. **Install all dependencies**  
   Run the following command in your terminal or Anaconda Prompt:

2. **Add your conversation dataset**  
Place your dataset (e.g., `conversationfile.csv` or `.xlsx`) in the same directory as the notebook.

3. **Run the notebook**  
Open and execute `ChatRec_Model.ipynb` step-by-step in Jupyter Notebook or Google Colab.  
The notebook will:
- Preprocess and clean your dataset  
- Fine-tune the GPT-2 model  
- Evaluate using BLEU, ROUGE, and Perplexity metrics  
- Save outputs both as a Hugging Face model and a serialized `.joblib` file  

----------------------------------------------------------
Model Access
----------------------------------------------------------
You can directly download the serialized fine-tuned model here:  
👉 **Model.joblib (Google Drive Link)**  
https://drive.google.com/file/d/1fT0YOlEB9pKVD5dTh5VPZZ8m2U0PLmvQ/view?usp=sharing  

> Note: You may need to sign in or request access depending on sharing permissions.

----------------------------------------------------------
How to Open and Inspect the `.joblib` File
----------------------------------------------------------
To load and explore the serialized model and tokenizer, follow these steps:

```python
import joblib

# Path to the downloaded .joblib file
model_path = "Model.joblib"

# Load the object
obj = joblib.load(model_path)

# Identify model and tokenizer
if isinstance(obj, dict):
 model = obj.get("model")
 tokenizer = obj.get("tokenizer")
elif hasattr(obj, "model") and hasattr(obj, "tokenizer"):
 model = obj.model
 tokenizer = obj.tokenizer
else:
 print("Unknown structure:", type(obj))
 print(obj)
 raise ValueError("Unexpected format in .joblib file")

# Test a prediction
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated reply:", reply)
