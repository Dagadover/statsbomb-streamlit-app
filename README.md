# MatchReport AI

Automated football match reports web application built with Python and Streamlit, using StatsBomb Open Data.

## Live App
https://statsbomb-matchreport-app.streamlit.app/

## Overview

**MatchReport AI** is a web application that automatically generates interactive football match reports from StatsBomb Open Data.  
The app allows users to explore competitions and matches, visualize key performance metrics, and export complete match reports in PDF and image formats.

This project demonstrates how data analysis workflows can be transformed into end-user products through interactive web applications.

## Features

- Select football competition and match dynamically  
- Interactive data visualizations  
- Automatic generation of match reports  
- Export reports as PNG and PDF  
- Download all generated outputs as a ZIP file  
- Runs fully in the browser  

## Tech Stack

- Python  
- Streamlit  
- StatsBomb Open Data  
- Pandas  
- Matplotlib  
- Plotly  

## Local Installation

Clone the repository and run the app locally:

```bash
git clone https://github.com/Dagadover/statsbomb-streamlit-app.git
cd statsbomb-streamlit-app

python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
streamlit run app.py

The app will be available at:
http://localhost:8501

Deployment

The application is deployed using Streamlit Community Cloud and is publicly accessible via the link above.

Data Source

All data is retrieved from StatsBomb Open Data, provided for educational and research purposes.

Author

Daniel Aguiñaga
Data Analyst / Data Scientist