# Webookcare

## Tested Environments

- **OS:** Debian-based distributions  
- **Python:** 3.10  

## Installation

### Step 1: Clone the Repository  
```bash
git clone https://github.com/arian47/webookcare.git
cd webookcare
```

### Step 2: Install in Editable Mode  
```bash
python3.10 -m pip install -e .
```

### Handling Dependency Issues  

If you encounter errors related to subprocesses failing to build dependencies, try upgrading them first:  
```bash
python3.10 -m pip install --upgrade pip setuptools wheel --break-system-packages
```

#### If you get errors upgrading `wheel`:  
1. **Manually remove it first:**  
   ```bash
   sudo apt remove python3-wheel
   ```
2. **Ensure `distutils` is installed if `pip` is inaccessible:**  
   ```bash
   sudo apt install python3.10-distutils
   curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
   ```
3. **Retry upgrading:**  
   ```bash
   python3.10 -m pip install --upgrade pip setuptools wheel --break-system-packages
   ```

After resolving these, reinstall the package:  
```bash
python3.10 -m pip install -e . --break-system-packages
```

## Setting Up Dependencies  

### Step 1: Execute `path_struct.py`  

After installation, a **secretly shared** file named `path_struct.py` must be executed to download necessary files and folders for data and model operations:  
```bash
python3.10 path_struct.py
```
- `path_struct.py` is located in the same directory as `setup.py` and `requirements.txt`.
- **Important:** URLs for folder sharing might reset, causing errors. Ensure you have updated links.

---

## Running the Server  

### Step 1: Install `uvicorn`  
```bash
python3.10 -m pip install uvicorn
```

### Step 2: Start the Server  
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Handling `cryptography` Package Issues  

If you face issues with the `cryptography` package:  

1. **Remove the OS-managed version:**  
   ```bash
   sudo apt-get remove --purge python3-cryptography
   ```
2. **Reinstall it using `pip`:**  
   ```bash
   python3.10 -m pip install cryptography
   ```
3. **Ensure OpenSSL development libraries are installed:**  
   ```bash
   sudo apt-get install libssl-dev
   ```

---

## Database Setup  

### Step 1: Verify MySQL Installation  
```bash
sudo systemctl status mysql
```
If not installed:  
```bash
sudo apt update
sudo apt install mysql-server
```

### Step 2: Restore MySQL Database  

If you have a **database dump**, follow these steps:

#### **(a) Transferring and Extracting Database Dump**  

1. **Transfer dump from another machine:**  
   ```bash
   scp database.tar.gz username@server_address:/path/to/destination
   ```
2. **Extract the dump file:**  
   ```bash
   tar -zxvf database.tar.gz
   ```

#### **(b) Importing MySQL Database**  

1. **Create a new database:**  
   ```sql
   CREATE DATABASE database;
   ```
2. **Import the dump into MySQL:**  
   ```bash
   mysql -u root -p database < /path/to/database.sql
   ```

### Handling MySQL Access Issues  

If you cannot access MySQL, update root user authentication:  
```sql
SELECT user, host, plugin FROM mysql.user WHERE user='root';
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'your_password';
FLUSH PRIVILEGES;
```

Or grant full privileges:  
```sql
GRANT ALL PRIVILEGES ON *.* TO 'root'@'localhost' WITH GRANT OPTION;
FLUSH PRIVILEGES;
```

---

## Running the Server After Database Setup  

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Usage  

Requests are sent to a host running the main **FastAPI** app.  
Currently, only the **patient ensemble model** has been implemented, which relies on sub-models:  

- **Care Services Recommender**
- **Credentials Recommender**
- **Distance Predictor**
- **Collaborative Filtering** (for ranking based on reviews)

### Setting Up the Server  

1. **Access saved models:** Ensure models or training data are available.
2. **Store database credentials:** Save credentials in a `.env` file.
3. **Install required dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```
4. **Launch FastAPI:**  
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
5. **Send API requests:** Queries can be made to the server.

---

## Text Classification Model  

The prediction is a weighted combination of multiple models for **careseekers** and **HCWs**, including:

- **Service & Credentials Recommender models:**  
  - Predicts labels using **n-gram techniques** (bigrams for services, unigrams for credentials).
  - Uses an **MLP model** due to limited data for sequence models.
  - **Text augmentation** was applied for training.

- **Location Predictor model:**  
  - Uses the **Haversine formula** for distance estimation.

- **Review Ranking model:**  
  - Implements **collaborative filtering** for recommendations.

### Text Augmentation Methods  

Several techniques have been used to improve text classification:

- **Manual sentence rewriting**
- **Open-source LLMs** (e.g., Ollama, GPT-based models)
- **Synonym replacement** (via NLTK)
- **Grammatical restructuring**
- **Paraphrasing tools** (e.g., QuillBot, TextRazor)
- **Back-translation** (Google Translate API, MarianMT)
- **Hugging Face Transformers** (e.g., BERT, GPT-3, T5, Pegasus)
- **Text augmentation libraries** (e.g., `nlpaug`, `textattack`)
- **Crowdsourcing paraphrasing** (Amazon Mechanical Turk)

---

## Optional
if you got more CPUs available you can create more workers as follows:
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000 --daemon
```
and you can also Set Up a Systemd Service for Auto-Restart cases to server start running after reboots
```bash
sudo nano /etc/systemd/system/fastapi.service
```

Paste the following (update paths as needed):
```bash
[Unit]
Description=FastAPI Server
After=network.target

[Service]
User=ubuntu  # Change to your Linux username
Group=ubuntu
WorkingDirectory=/path/to/your/project  # Update this to your app directory
ExecStart=/usr/local/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:6000
Restart=always  # Auto-restart on failure
RestartSec=5  # Wait 5 sec before restarting

[Install]
WantedBy=multi-user.target
```

Enable & Start the Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable fastapi
sudo systemctl start fastapi
```
Check Server Status
```bash
sudo systemctl status fastapi
```

Restart Server Manually (if needed)
```bash
sudo systemctl restart fastapi
```

## Contribution  

Feel free to contribute, open issues, or suggest improvements!

overfitting is clearly visible even with the basic MLP structure (consisting of dense(8192), dropout(.3), dense(4096), dropout(.5), dense(len(multi label vocabulary))

![int 1 gram MLP train data, bigram multihot labels](https://github.com/user-attachments/assets/63bf8ca2-8c37-4249-91b8-e2881d66622c)

![MLP int 2 gram train data, bigram multihot labels](https://github.com/user-attachments/assets/f2f72994-20c1-495a-b477-6d49557c68af)

