---
title: LanggraphAgenticAI
emoji: 🐨
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.42.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Refined langgraphAgenticAI
---

# LanggraphAgenticAI 🐨

LanggraphAgenticAI is a refined AI chatbot application built using Streamlit. It leverages advanced language models and graph-based state management to provide intelligent responses and support.

## Features ✨

- **Streamlit Integration**: Easy-to-use web interface powered by Streamlit.
- **Advanced Language Models**: Utilizes state-of-the-art language models for generating responses.
- **Graph-Based State Management**: Manages conversation state using a graph-based approach.
- **Document Retrieval**: Supports retrieval of documents from URLs and PDFs.
- **Human Support Routing**: Routes queries to human support when necessary.

## Installation 🛠️

1. **Clone the repository**:
    ```sh
    git clone https://github.com/HardikNegi9/EvalAIChatBot.git
    cd EvalAIChatBot
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment**:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

5. **Set up environment variables**:
    Create a [.env](http://_vscodecontentref_/1) file in the root directory and add the following:
    ```env
    GROQ_API_KEY="your_groq_api_key"
    GOOGLE_API_KEY="your_google_api_key"
    HF_TOKEN="your_huggingface_token"
    ```

## Usage 🚀

1. **Run the Streamlit app**:
    ```sh
    streamlit run app.py
    ```

2. **Open your browser** and navigate to `http://localhost:8501` to interact with the LanggraphAgenticAI chatbot.

## Configuration ⚙️

The application configuration is managed using an INI file located at [uiconfigfile.ini](http://_vscodecontentref_/2). You can update the model options and page title as needed.

## Contributing 🤝

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License 📄

This project is licensed under the Apache-2.0 License. See the LICENSE file for details.

## Contact 📧

For any inquiries or support, please contact [hardiknegi06@gmail.com](mailto:hardiknegi06@gmail.com).

---

Thank you for using LanggraphAgenticAI! We hope it helps you achieve your goals. 😊

