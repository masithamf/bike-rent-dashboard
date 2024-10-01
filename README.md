# Cragoo Bike Rental Service Dashboard &#x2728;
This is an interactive way to preview the dataset Bike Sharing based on Python

## Setup Environment - Anaconda
```
conda create --name main-ds python=3.9
conda activate main-ds
pip install -r requirements.txt
```
## Setup Python Environment - Terminal
```
cd project_submission
python -m venv .venv

# Windows command prompt
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS and Linux
source .venv/bin/activate

pip install streamlit
pip install -r requirements.txt
```
> Whenever you want to use your environment, go to your project folder where `.venv` directory exist and activate it

## Run Streamlit App
```
streamlit run dashboard.py
```
