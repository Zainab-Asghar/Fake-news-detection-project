 1: Fake News Detection AI

This project detects whether a news article is **Fake or Real** using a fine-tuned **DistilBERT** model.

2: Features
- Transformer-based fake news detection
- Cleaned & balanced dataset
- Flask API for predictions
- Ready for deployment

3: Model
- DistilBERT (fine-tuned)
- Accuracy: ~99%
- Download Pretrained Model & Dataset

 **Pretrained Model:** [Download here](https://drive.google.com/drive/folders/1t-g0FP_zOKJ4v5KVtM13bxgv9jHwYE5C?usp=sharing)  
- **Dataset:** [Download here](https://drive.google.com/drive/folders/1qrk2gRanJGubQyqwc3AJf28Atzip-vv3?usp=sharing)

> After downloading, place the model in the `models/` folder and the dataset in the `data/` folder.

4: Tech Stack
- Python
- Hugging Face Transformers
- PyTorch
- Flask
- Scikit-learn

5: Run Locally

```bash
pip install -r requirements.txt
python app.py
