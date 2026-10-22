# ai-data-analysis-agent
# 🤖 AI Data Analysis Agent

An AI-powered data analysis application built with Python, Streamlit, Pandas, DuckDB, Plotly, and Google Gemini.

The goal of this project is to make data analysis easier through an interactive interface where users can explore datasets, perform analysis, visualize results, and ask questions using AI.

## ✨ Features

- 📂 Upload and explore CSV and Excel datasets
- 📊 Generate statistical summaries and data insights
- 🧹 Check missing values and duplicate records
- 📈 Create interactive data visualizations
- 🧮 Query datasets using SQL
- 🤖 Ask data-related questions using Google Gemini
- 📄 Review and download analysis reports

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Streamlit | Interactive user interface |
| Pandas & NumPy | Data processing and analysis |
| Plotly | Interactive visualizations |
| DuckDB | SQL-based data analysis |
| Google Gemini API | AI-powered question answering |

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/vaishnavijoshi438-oss/ai-data-analysis-agent.git
cd ai-data-analysis-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Create a `.env` file in the project directory:

```text
GEMINI_API_KEY=your_gemini_api_key
```

Replace the example value with your own Gemini API key.

**Security:** Never commit your `.env` file or expose API keys publicly.

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📁 Project Structure

```text
ai-data-analysis-agent/
├── app.py
├── requirements.txt
├── test_gemini.py
├── .gitignore
└── README.md
```

## 🎯 Project Objective

To build an interactive AI-assisted data analysis tool that helps users explore datasets, understand patterns, and obtain useful insights with less manual effort.

## 🔮 Future Improvements

- Support for additional file formats and larger datasets
- More advanced AI-driven analytical workflows
- Improved data visualizations and reporting
- Enhanced error handling and data validation
- Additional automated data quality checks

## 👩‍💻 Author

**Vaishnavi Joshi**

GitHub: [@vaishnavijoshi438-oss](https://github.com/vaishnavijoshi438-oss)

---

⭐ If you find this project interesting, feel free to explore the repository and share feedback.
