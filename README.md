# 🔥 Fine-Tuning LLaMA 2 on Custom Wildfire Dataset

This project demonstrates how to fine-tune the **LLaMA 2 7B Chat** model using a **custom text dataset** about Hawaii wildfires. It uses **LoRA (Low-Rank Adaptation)** and **4-bit quantization** to fine-tune the model efficiently, even on limited hardware like Google Colab.

---

## 📌 Objective

- Fine-tune a pre-trained Large Language Model (LLM) to better understand wildfire-related questions.
- Use real-world wildfire text files to improve the model’s domain-specific accuracy.
- Apply memory-efficient techniques for fast, lightweight training.

---

## 🛠️ Technologies Used

| Tool/Library       | Purpose                                         |
|--------------------|-------------------------------------------------|
| Transformers       | Load and fine-tune LLaMA 2                      |
| PEFT (LoRA)        | Lightweight training via Low-Rank Adaptation    |
| BitsAndBytes       | 4-bit quantization for memory optimization      |
| Datasets           | Load and prepare custom text data               |
| Accelerate         | Speed up training performance                   |
| Huggingface Hub    | Model access and authentication                 |
| Google Colab (GPU) | Training and inference environment              |


## 🚀 Step-by-Step Workflow

### 1️⃣ Install Required Libraries  
Install all necessary packages using pip.

### 2️⃣ Check GPU Availability  
Ensure CUDA is enabled for faster model training.

### 3️⃣ Login to Hugging Face  
Authenticate to access LLaMA 2 using `notebook_login()`.

### 4️⃣ Load LLaMA 2 Model (4-bit)  
Load the pre-trained model using `nf4` 4-bit quantization.

### 5️⃣ Clone and Read Dataset  
Clone wildfire data from GitHub and read the first line from each file as training input.

### 6️⃣ Tokenize Text Data  
Tokenize the input text using the LLaMA tokenizer and add special tokens if needed.

### 7️⃣ Apply LoRA Configuration  
Enable LoRA to train only specific model layers, reducing memory usage.

### 8️⃣ Fine-Tune the Model  
Use Hugging Face `Trainer` for lightweight training. Results are saved in `finetunedModel/`.

### 9️⃣ Run Evaluation  
Ask questions like:
- *“When did Hawaii wildfires start?”*
- *“How many acres were burned in the Lahaina fire?”*

---

## ✅ Example Results
Q: When did Hawaii wildfires start?
A: The wildfires in Hawaii started on August 9, 2023.

Q: How many acres were burned in the Lahaina fire?
A: The Lahaina fire burned approximately 3,000 acres of land.


## 📌 Key Highlights

- ✅ Uses real data for task-specific fine-tuning  
- 🔄 Fast, low-memory training using LoRA  
- ⚡ Quantized model runs on smaller GPUs (even Colab)  
- 🧪 Fine-tuned for Q&A-style use cases  

---

## 🔗 References

- [LLaMA 2 on Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)  
- [LoRA Research Paper](https://arxiv.org/abs/2106.09685)  
- [Fine-Tuning LLMs Dataset GitHub](https://github.com/poloclub/Fine-tuning-LLMs)  
- [Transformers Documentation](https://huggingface.co/docs/transformers)

## 🙋 Project Owner

Created and maintained by **`hari927`**  
Feel free to reach out anytime to discuss ideas, ask questions, or report issues.  
You can also [open an issue](https://github.com/hari927/fine-Tuning-llm-using-custom_dataset/issues) on this repository.
