# 654321
# SimpleLLM & SAST Blackboard

This repository contains a **scientific programming prototype** used to explore two main concepts:

1. **SimpleLLM Architecture** – a minimalistic framework for semantic typing.
2. **SAST Blackboard** – A multi-agent approach for semantic typing with focus of decision understanding via signals.

The goal of this project is to provide a clean and modifiable structure for testing, benchmarking, and extending LLM-driven analysis workflows.

---

## Project Structure



---

## Environment Setup

Create in Root folder if this project a new folder named "env", in which you must create a file named **`.env`** containing:
OPENAIKEY="your api key here"


This environment variable is required for all LLM-based components.

> **Note:**  
> Other parameters which will be used for both architectures (Sample IDs, historical references etc.) are located in `main.py` and can be modified easily.

---

## Running the Code

1. Make sure you have Python ≥ 3.10 installed.
2. Install the dependencies (requirements.txt):
3. Ensure that `env/.env` exists and contains your API key.
4. Start the program main.py: