# Tech Stack Recommender

A Python-based AI recommendation system that suggests suitable technology career paths based on a user's existing skills. The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** and **Cosine Similarity** to compare a user's skill set against predefined career profiles and recommend the best matches.

---

## 📌 Overview

Choosing the right technology career can be challenging, especially for beginners. This project demonstrates how Information Retrieval and Machine Learning concepts can be applied to build a simple recommendation engine.

The program:

* Reads career roles and required skills from a CSV dataset.
* Converts skills into TF-IDF vectors.
* Calculates similarity using Cosine Similarity.
* Recommends the top three most suitable career paths.
* Handles the Cold Start Problem by providing a default baseline when insufficient user skills are entered.

---

## 🚀 Features

* Career recommendation based on user skills
* CSV dataset ingestion
* Shared vocabulary generation
* TF (Term Frequency) calculation
* IDF (Inverse Document Frequency) calculation
* TF-IDF vector generation
* Cosine Similarity-based ranking
* Top-3 recommendation system
* Cold Start Problem handling
* Uses only Python standard libraries

---

## 🧠 Concepts Used

* Information Retrieval
* Feature Extraction
* TF (Term Frequency)
* IDF (Inverse Document Frequency)
* TF-IDF
* Vector Space Model
* Cosine Similarity
* Recommendation Systems

---

## 📂 Project Structure

```
Project-3-Tech-Stack-Recommender/
│
├── tech_stack_recommender.py     # Main Python program
├── raw_skills.csv                # Dataset containing roles and skills
├── README.md                     # Project documentation
├── .gitignore                    # Git ignore rules
└── requirements.txt              # Optional (no external libraries required)
```

---

## ⚙️ Requirements

* Python 3.8 or above

No external Python libraries are required.

The project uses only built-in modules:

* csv
* math
* collections

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git
```

### 2. Navigate to the project folder

```bash
cd Project-3-Tech-Stack-Recommender
```

### 3. Make sure the dataset is present

Ensure that `raw_skills.csv` is in the same directory as `tech_stack_recommender.py`.

### 4. Run the program

```bash
python tech_stack_recommender.py
```

---

## 💻 Example

### Input

```
Python, SQL, Automation
```

### Output

```
--- TOP 3 RECOMMENDED CAREER PATHS ---

1. Data Analyst               | Match Score: 92.4%
2. Machine Learning Engineer  | Match Score: 88.7%
3. Backend Developer          | Match Score: 84.3%
```

*(Actual results depend on the dataset.)*

---

## 🔍 How It Works

1. **Dataset Ingestion**

   * Reads career roles and their associated skills from the CSV file.

2. **Vocabulary Construction**

   * Creates a shared vocabulary containing every unique skill.

3. **TF-IDF Vectorization**

   * Calculates TF values.
   * Computes IDF values.
   * Builds weighted TF-IDF vectors.

4. **Similarity Calculation**

   * Computes Cosine Similarity between the user's skills and every career profile.

5. **Ranking**

   * Sorts careers based on similarity scores.
   * Returns the Top 3 recommendations.

---

## ❄️ Cold Start Handling

If the user enters fewer than three skills, the system automatically supplements the input with a default baseline:

* Python
* Automation
* SQL

This prevents zero-vector comparisons and improves recommendation quality.

---

## 📈 Future Improvements

* Graphical User Interface (Tkinter)
* Streamlit web application
* Flask REST API
* Larger and more diverse dataset
* Skill weighting based on experience level
* Learning-based recommendation engine
* Interactive visualizations
* Personalized career roadmap generation

---

## 📚 Learning Outcomes

This project demonstrates practical implementation of:

* Information Retrieval
* Recommendation Systems
* Vector Space Models
* Feature Engineering
* Cosine Similarity
* Python File Handling
* Data Processing

---

## 👨‍💻 Author

**Kashvi**

Computer Science Undergraduate

---

## 📄 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it for educational purposes.
