# 🎓 College Review Rating Prediction Model
This project uses Machine Learning and Natural Language Processing (NLP) to predict numerical ratings (e.g., out of 10) based on college review texts submitted by students.

🧠 Project Objective
Given a text review of a college (like “The infrastructure is good, but placements are poor.”), the model predicts the expected rating (e.g., 6.8 / 10). It helps analyze student feedback automatically and can be used by college review platforms.

🔍 Features
Clean and preprocess college reviews

Convert text to numerical features using TF-IDF

Predict rating using a Regression Model

Supports integration with web apps (Gradio/Streamlit ready)

Ready for deployment and testing

🧰 Tech Stack
Python

scikit-learn

NLTK

Pandas

NumPy

TF-IDF Vectorizer

(Optional: Gradio or Streamlit for UI)

📁 Folder Structure
bash
Copy
Edit
LIve-Model/
│
├── model.pkl             # Trained regression model
├── tfidf.pkl             # Saved TF-IDF vectorizer
├── preprocess.py         # Text preprocessing functions
├── app.py                # Optional UI (Gradio or Streamlit)
├── notebook.ipynb        # Jupyter Notebook for training/testing
├── requirements.txt      # List of required packages
└── README.md             # Project overview
🚀 How to Run
Clone the repository

bash
Copy
Edit
git clone https://github.com/shubham15kumarr/LIve-Model.git
cd LIve-Model
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Notebook or App

bash
Copy
Edit
# To run the notebook (for training/testing)
jupyter notebook

# OR to run the Gradio/Streamlit app
python app.py
🧪 Example
Input Review:

"Campus is well maintained, but faculty support is average."

Predicted Rating:

7.2 / 10

📦 Model Training Details
Vectorizer: TF-IDF

Stopwords Removal: Using NLTK

Regression Model: Linear Regression (or specify if different)

Evaluation Metric: R² Score

✍️ Author
Shubham Kumar

⭐ Contribute
Feel free to fork the project, raise issues, or suggest improvements!

📜 License
This project is open-source and available under the MIT License.

📌 Notes
Add your model.pkl and tfidf.pkl files in the root directory before running predictions.

You can enhance this with more data and deep learning models in the future.LIve-Model
