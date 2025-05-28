# 📝 Arabic Part-of-Speech Tagging using BERT

This project performs **Part-of-Speech (POS) tagging for Arabic text** using a fine-tuned BERT model from Hugging Face's Transformers library. It leverages the `asafaya/bert-base-arabic` pre-trained model for token classification.


## 📌 Objective

The main goal of this project is to:

* Perform accurate **POS tagging** for Arabic sentences.
* Fine-tune or utilize a pre-trained transformer model for sequence labeling.
* Demonstrate the use of Hugging Face Transformers with TensorFlow.

---

## 🧠 Model Details

The model used is:

```
asafaya/bert-base-arabic
```

This model was fine-tuned for **token classification** to predict POS tags.

---

## 🛠 How It Works

### 🔹 Step 1: Load Tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
```

### 🔹 Step 2: Load Model for Token Classification

```python
from transformers import TFAutoModelForTokenClassification

model = TFAutoModelForTokenClassification.from_pretrained(
    "asafaya/bert-base-arabic",
    num_labels=len(le.classes_),        # Number of POS tags
    id2label=id2label,                  # Mapping from ID to label (e.g., {0: "NOUN", 1: "VERB", ...})
    label2id=label2id                   # Mapping from label to ID (e.g., {"NOUN": 0, "VERB": 1, ...})
)
```

---

## 🗂 Label Mappings

Make sure you define your POS label mappings based on your dataset:

```python
label2id = {"NOUN": 0, "VERB": 1, "ADJ": 2, ...}
id2label = {0: "NOUN", 1: "VERB", 2: "ADJ", ...}
```

You can use `sklearn.preprocessing.LabelEncoder` to create `le.classes_` if you're encoding the labels from your dataset.

---

## 📋 Input Format

The input should be a **tokenized Arabic sentence**:

```python
sentence = "أنا أحب البرمجة"
tokens = tokenizer(sentence, return_tensors="tf", truncation=True, is_split_into_words=False)
```

---

## ✅ Output Format

The model will output a list of predicted POS tags aligned with the tokens:

```python
outputs = model(tokens)
predictions = tf.argmax(outputs.logits, axis=-1)
```

Post-process these predictions to map back to readable POS tags using `id2label`.

---

## 🧪 Dependencies

* Python 3.7+
* TensorFlow
* Transformers (`pip install transformers`)
* scikit-learn (for label encoding)

---

## 📂 Folder Structure

```
arabic-pos-tagging/
│
├── data/                  # Dataset used for training/evaluation
├── model/                 # Saved model weights (if any)
├── pos_tagger.py          # Core script
├── utils.py               # Label mapping, preprocessing utilities
└── README.md              # Project documentation
```

---

## 📌 Use Cases

* Linguistic analysis of Arabic text
* NLP preprocessing for Arabic models
* Educational tools and grammar checkers

---

## 🙌 Credits

* Pre-trained model: [asafaya/bert-base-arabic](https://huggingface.co/asafaya/bert-base-arabic)
* Hugging Face Transformers

---

Let me know if you'd like an example notebook or script for inference!
