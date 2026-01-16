# Deploying HybridSVD for Web-Based Restaurant Recommendations

**Bachelor Thesis Project** **Delft Center for Systems and Control (DCSC) | TU Delft**

This repository contains the source code, datasets, and documentation developed for the Bachelor thesis: *"Deploying HybridSVD for Web-Based Restaurant Recommendations"*.

The project focuses on implementing a HybridSVD recommender system that integrates collaborative filtering with content-based features, deployed via a web-based interface for real-time user interaction.

---

## 📂 Repository Contents

This project is divided into four main components:

### 1. HybridSVD Recommender System
Implementation of the Hybrid Singular Value Decomposition (SVD) algorithm. This module handles:
* Matrix factorization of user-item interactions.
* Integration of side information (metadata) to solve the cold-start problem.
* Popularity bias mitigation using scaling techniques.

### 2. Google Local Businesses Data Retriever
A Python-based tool (`gmapID_retriever_GoogleAPI.py`) designed to enrich the dataset.
* Fetches official Google Place IDs, GPS coordinates, and editorial summaries.
* Downloads high-quality images for restaurants using the Google Places API.

### 3. Data Preprocessor
A suite of scripts to clean and transform raw data into the format required by HybridSVD.
* **Filtering:** Applies a recursive k-core filter (e.g., 20-core) to ensure data density.
* **Binarization:** Converts explicit ratings into implicit feedback (0/1) for interaction modeling.
* **Similarity Matrix Construction:** Generates item-item similarity matrices using:
    * **Geolocation:** Haversine distance calculations.
    * **Categories:** Overlap coefficients.
    * **Descriptions:** Semantic embeddings using sBERT (Sentence-BERT).

### 4. Web Server & Interface
A web application serving the recommendations to end-users.
* Backend logic to process user input and generate recommendations in real-time.
* Frontend interface for displaying restaurant details and photos.

---

## 📖 Documentation & User Manuals

Detailed instructions on how to install, configure, and run each component can be found in the **Manuals** folder.

* **[Link to User Manuals](./manuals)**

---

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scipy, Scikit-learn
* **Deep Learning:** sBERT (Sentence Transformers)
* **API:** Google Places API
* **Web Framework:** Flask (or similar, adjust based on your code)

---

**Institution:** Delft University of Technology (TU Delft)

---
