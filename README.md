# Quantellix: Autonomous AI Supply Chain & Order Fulfillment Platform

Quantellix is an end-to-end intelligent supply chain and order fulfillment orchestration platform powered by FastAPI, modern analytics dashboards, autonomous Gemini AI agents, and machine learning pipelines (Sales Forecasting, Demand Planning, and Multi-Vendor Procurement).

---

## 🚀 Quick Start (Sharing with Friends / Team)

### ⚠️ Important Rule for Sharing
When sharing this project folder (via ZIP, Google Drive, USB, or Git):
- **DO NOT include the `.venv` folder**: Virtual environments contain binary paths unique to your specific machine and operating system.
- Your friend's machine will generate its own clean `.venv` automatically using the 1-click script below.

---

### Option 1: 1-Click Automated Setup (Windows) — Recommended
If your friend is on Windows, setup requires **zero manual command typing**:

1. **Unzip or place the project folder** anywhere on the computer.
2. **Double-click `setup_and_run.bat`**:
   - It automatically checks for Python 3.10+.
   - Creates a fresh `.venv` virtual environment.
   - Installs all dependencies from `requirements.txt`.
   - Creates `.env` from `.env.example`.
   - Launches the portal server and opens `http://127.0.0.1:5000/portal/marketplace` directly in the default browser.
3. For subsequent daily runs, simply double-click `run_portal.bat`.

---

### Option 2: 1-Click Automated Setup (macOS / Linux)
If your friend is on a Mac or Linux machine:

1. Open a terminal inside the project folder:
   ```bash
   chmod +x run_portal.sh
   ./run_portal.sh
   ```
2. The script will automatically create `.venv`, install packages, and boot the server at `http://127.0.0.1:5000`.

---

### Option 3: Manual Step-by-Step Setup (Any OS)

If you prefer to run the setup manually in the terminal:

#### 1. Prerequisites
- **Python 3.10 - 3.13** installed on your system ([Download Python](https://www.python.org/downloads/)).
- *(Windows users: Check the box "Add Python to PATH" during installation).*

#### 2. Open Terminal in Project Directory
Navigate to the root folder where `requirements.txt` is located.

#### 3. Create a Virtual Environment (`.venv`)
```bash
# Windows
python -m venv .venv

# macOS / Linux
python3 -m venv .venv
```

#### 4. Activate the Virtual Environment
```bash
# Windows (Command Prompt - CMD)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Note: If PowerShell gives an execution policy error, run:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# macOS / Linux
source .venv/bin/activate
```
*(Your terminal prompt will now display `(.venv)` in front of the path).*

#### 5. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 6. Configure Environment Variables (`.env`)
Make sure a `.env` file exists in the root directory. If not, copy it from `.env.example`:
```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```
Open `.env` in any text editor and ensure your Google Gemini API key is configured:
```env
GEMINI_API_KEY=your_actual_gemini_api_key
```
*(You can obtain a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey)).*

#### 7. Start the Application Server
```bash
python -m uvicorn portal.server:app --host 127.0.0.1 --port 5000 --reload
```

#### 8. Access the Portal
Open your browser and navigate to:
👉 **[http://127.0.0.1:5000/portal/marketplace](http://127.0.0.1:5000/portal/marketplace)**

---

## 📁 Project Architecture & Key Folders

```
order fulfillment/
│
├── .venv/                      # Local Python virtual environment (auto-created)
├── .env                        # Local environment variables & Gemini API key
├── .env.example                # Template for environment variables
├── requirements.txt            # All required Python packages
├── setup_and_run.bat           # 1-Click automated setup & launcher for Windows
├── run_portal.bat              # Daily 1-click launcher for Windows
├── run_portal.sh               # 1-Click automated setup & launcher for Mac/Linux
├── README.md                   # Complete documentation and setup guide
│
├── portal/                     # Web Portal Application
│   ├── server.py               # FastAPI application entry point & API routes
│   ├── database.py             # SQLite DB models & transaction management
│   ├── eda_engine.py           # Automated EDA & statistical profiling engine
│   ├── gemini_service.py       # Autonomous LLM Chatbot service (Google GenAI)
│   ├── static/                 # CSS stylesheets, JavaScript files, images
│   └── templates/              # Jinja2 HTML templates
│       ├── marketplace.html    # Quantellix App & Problem Marketplace
│       ├── chatbot.html        # Interactive AI Chatbot & Agent interface
│       ├── executive_dashboard.html # S1: Sales Forecasting Dashboard
│       ├── demand_planning_dashboard.html # S2: Demand Planning Dashboard
│       └── procurement_dashboard.html     # S3: Procurement & Vendor Dashboard
│
├── src/                        # Data Science & Machine Learning Pipelines
│   ├── demand_planner.py       # S&OP Consensus & Safety Stock logic
│   ├── procurement_agent.py    # Multi-vendor MCDA procurement allocation
│   ├── prophet_pipeline.py     # Facebook Prophet forecasting model
│   ├── sarima_pipeline.py      # SARIMA statistical forecasting model
│   └── lstm_pipeline.py        # PyTorch LSTM deep learning pipeline
│
└── artifacts/                  # Generated plots, reports, and exported datasets
```

---

## 💡 Troubleshooting & FAQ

- **Port 5000 is already in use:**
  If another program uses port 5000, start the server on another port:
  ```bash
  python -m uvicorn portal.server:app --host 127.0.0.1 --port 8000 --reload
  ```
  Then visit `http://127.0.0.1:8000/portal/marketplace`.

- **PowerShell says "Execution of scripts is disabled":**
  Run:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```
  and then run `.venv\Scripts\Activate.ps1`.

- **Chatbot says "AI Service temporarily unavailable":**
  Verify that your `.env` contains a valid `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/app/apikey).
