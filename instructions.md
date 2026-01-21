# Project 2: Image-to-Text Generation and Analysis Platform

A hybrid AI application that analyzes images to extract text, describe scenes, and auto-structure data (tables/mind map). It uses a *Local Brain* (OCR + Ollama) for lightweight tasks and a *Cloud Brain* (Google Colab VLM) for heavy visual reasoning.

## Specifications

* **Frontend:** HTML5, CSS3, JavaScript
* **Backend:** Python (FastAPI)
* **OCR Engine:** EasyOCR (Runs locally on CPU)
* **Fallback LLM:** Ollama (`qwen2.5:1.5b` or similar) - Runs locally
* **Primary VLM:** `Qwen2.5-VL-3B-Instruct` - Runs on Google Colab (T4 GPU)
* **Tunneling:** ngrok (Connects local backend to Colab)
* **Storage:** Local filesystem (`/uploads` folder)

## Project Structure

project-folder/
├── backend/
│   ├── config.py               # *Update ngrok URL here*
│   ├── describe_module.py
│   ├── main.py                 # *Entry point*
│   ├── ocr_module.py
│   ├── ollama_client.py
│   ├── structure_module.py
│   └── utils.py
├── uploads/                   # *Images saved here automatically*
├── index.html                 # *Frontend UI*
├── instructions.md
├── README.md
├── requirements.txt           # *Contains all dependencies*
└── vlm.ipynb                  # *Upload this to Google Colab (preferably use the file shared in Google Drive)* 

## Execution Steps

### Phase 1: Local Setup

1. **Prepare Environment:**
* Download the project folder from Google Drive (*https://drive.google.com/drive/folders/17SSANRdUZcJTq3Ib3srfjqlq7MIr2l78*) or clone the repository from GitHub (*https://github.com/rks746/image-to-text-generation.git*).
* Open the folder in your IDE (VS Code, PyCharm, etc.).
* Create a virtual environment (optional but recommended).


2. **Install Python Dependencies:**
Open your terminal in the project root directory and run:
*pip install -r requirements.txt*


3. **Setup Local AI (Ollama):**
* Download and install Ollama.
* Pull the text model used for the fallback mechanism (OCR analysis). Run this in your terminal: *ollama pull qwen2.5:1.5b*
* **Keep Ollama running** in the background (ensure the tray icon is active).


### Phase 2: Cloud Brain Setup (Google Colab)

4. **Open Colab:**
* Upload the *vlm.ipynb* file to Google Colab.
* **Crucial:** Go to **Runtime -> Change runtime type** and select **T4 GPU**.


5. **Configure ngrok:**
* Create a free account at ngrok.com.
* Go to **Dashboard -> Your Authtoken** and copy the token.
* In Colab, look for the **Secrets** (key icon) on the left sidebar.
* Add a new secret:
* Name: *NGROK_AUTH_TOKEN*
* Value: *(Paste your token)*
* Toggle "Notebook access" to **On**.


6. **Run the VLM:**
* Run all cells in the Colab notebook.
* Wait for the model to download and load (~2-3 minutes).
* The last cell will output a public URL (e.g., *https://2843-xx-xx.ngrok-free.app*). **Copy this URL.**


### Phase 3: Connecting & Running

7. **Update Config:**
* Open *backend/config.py* in your local IDE.
* Paste the ngrok URL into the *VLM_BASE_URL* variable.
* *Note: You must repeat this step every time you restart the Colab runtime.*


8. **Start Local Backend:**
Run the following command from the project root folder:
*uvicorn backend.main:app --reload*
*(If your terminal is already inside the *backend* folder, use *uvicorn main:app --reload instead).*


9. **Launch App:**
* Open your browser and go to: *http://127.0.0.1:8000/*
* Upload an image and click **Run**.
