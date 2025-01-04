# Quantum XOR Classifier Demo

This repository showcases a quantum-classical hybrid approach to classify XOR data using **Qiskit Machine Learning** and **Streamlit**.  
You’ll find:

- **QuantumXORClassifier**: A Python class that constructs and trains a QNN-based classifier.  
- **QuantumXORApp**: A Streamlit-based UI class for an interactive user experience, complete with circuit visualization and performance charts.

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Setup](#setup)  
3. [Usage](#usage)  
4. [Features](#features)  
5. [Troubleshooting](#troubleshooting)  
6. [License](#license)  
7. [Full Code](#full-code)

---

## Prerequisites

1. **Python 3.7+**  
2. [Qiskit and Qiskit Machine Learning](https://qiskit.org/documentation/machine-learning/).  
3. **Streamlit** for the web-based UI.  
4. **Matplotlib** for circuit visualization.  
5. **Altair** for interactive charts (or Plotly if you prefer a different charting library).  

All necessary dependencies are listed in **requirements.txt**.

---

## Setup

1. **Clone** or **download** this repository.  

2. **Create** and **activate** a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   # or on Windows:
   venv\Scripts\activate
   
Install required libraries:
```
pip install -r requirements.txt
```
Verify installation:
```
python -m pip list
```


Ensure that all the listed dependencies (Qiskit, Streamlit, etc.) are present.

## Usage

From the project’s root directory, run the Streamlit app:
```
streamlit run app/app.py
```


Replace quantum_xor_app.py with the actual filename containing your OOP code if it differs.
Open your web browser to the URL displayed in the terminal (usually http://localhost:8501).
On the sidebar, configure:

**Test Size Ratio**

**COBYLA Max Iterations**

**Random Seed**

**TwoLocal Repetitions**

**Entanglement Pattern**

**Shots**

Click Train Model to build and train the classifier. A progress bar will indicate the training steps.
Results (accuracy, classification report) appear on the main page, along with:
Quantum Circuit Diagram (within an expander)
Interactive Plots of Actual vs. Predicted labels.



## Features

Quantum-Classical Hybrid: Uses Qiskit’s EstimatorQNN for a parameterized quantum circuit.
Configurable Circuit: Easily tune reps, entanglement, and shots for different experiments.
Progress Bar: Watch your training steps in real time via Streamlit.
Circuit Visualization: Embedded Matplotlib diagram of the exact quantum circuit you’re training.
Interactive Charts: Leverages Altair for dynamic scatter plots of test data, color- and shape-coded by labels.

## Troubleshooting

### Missing ScriptRunContext Warnings
- Make sure you start the app with `streamlit run <filename>.py`.
- Running the script with just `python <filename>.py` can cause these warnings.

### Circuit Diagram Not Showing
- Check you have `matplotlib` installed.
- Confirm you’re calling `st.pyplot(fig)` (or `st.pyplot(fig_circuit)`) and not running into import issues.

### Dependency Conflicts
- Double-check `requirements.txt`.
- Use a dedicated virtual environment to avoid library version conflicts.

### Firewall/Proxy Issues
- If your environment blocks external sites, ensure you’re not attempting to load external images or resources.


