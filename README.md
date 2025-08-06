# LangChain Mastery with Ollama & OpenAI API

Based on James Calam's 5-hr LangChain course, this project is a comprehensive hands-on exploration of advanced Generative AI techniques and full-stack LLM workflows. It reflects my experience leveraging state-of-the-art frameworks and tools to build robust, scalable, and feature-rich AI applications.

## Highlights

- **LangChain Ecosystem**: Extensive use of the LangChain library and LangSmith for building, orchestrating, and debugging complex LLM pipelines.
- **Prompt Engineering**: Practical experience designing, refining, and evaluating prompts to achieve nuanced model behaviors and reliable outputs.
- **LLM Workflows**: Built asynchronous, streaming LLM pipelines with strong conversational memory and robust context handling for real-world applications.
- **Full Stack ReAct Agent Application**: Developed an end-to-end agent system using React, Tailwind CSS, NextJS (frontend), and FastAPI with Pydantic (backend). Includes reasoning checkpoints, extended web search, and modular agent tools.
- **API Integrations**: Integrated multiple APIs such as OpenAI, Ollama (for running open-source LLMs locally), and SerpAPI (for advanced web search).
- **Open Source LLMs**: Experimented with and deployed models like Llama 3.2:1b, Llama 3.2:3b, and Llama 3.3:70b via Ollama for flexible, high-quality inference.

## Project Structure

- `/api`: FastAPI backend with streaming endpoints
- `/app`: Next.js + React frontend with Tailwind CSS
- `/notebooks`: Jupyter notebooks for experimentation and testing
- `streaming-test.ipynb`: Quickstart notebook for streaming LLM output

## Running the Project

### Python Environment

The Python packages are managed using the [uv](https://github.com/astral-sh/uv) package manager, and so we must install `uv` as a prerequisite for the course. We do so by following the [installation guide](https://docs.astral.sh/uv/#getting-started). For Mac users, as of 22 Oct 2024 you can just enter the following in your terminal:
1. Install the [uv](https://github.com/astral-sh/uv) package manager:
    ```sh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2. Once `uv` is installed and available in your terminal, navigate to the root directory and execute the following:
    ```sh
    uv python install 3.12.7
    uv venv --python 3.12.7
    uv sync
    ```

> ❗️ You may need to restart the terminal if the `uv` command is not recognized by your terminal.

#### Using Venv in VS Code / Cursor

To use our new venv in VS Code or Cursor we simply execute:

```
cd example-chapter
cursor .  # run via Cursor
code .    # run via VS Code
```

This command will open a new code window, from here you open the relevant files (like Jupyter notebook files), click on the top-right **Select Environment**, click **Python Environments...**, and choose the top `.venv` environment provided.

#### Uninstalling Venvs

Naturally, we might not want to keep all of these venvs clogging up the memory on our system, so after completing the course we recommend removing the venv with:

```
deactivate
rm -rf .venv -r
```

### Ollama Setup

- [Download and install Ollama](https://ollama.com/) for local LLM inference.
- Pull the desired Llama models, e.g.:
    ```sh
    ollama pull llama3.2:3b
    ```
- Ensure Ollama is running by executing `ollama serve` in your terminal or running the Ollama application.
- Take note of the server port, by default Ollama runs on `http://localhost:11434`

### Backend API

- Navigate to `/api` and start the FastAPI server:
    ```sh
    uv run uvicorn main:app --reload
    ```
- API docs are available at `http://localhost:8000/docs`
- Streaming test notebook: `streaming-test.ipynb`

### Frontend App

- Navigate to `/app` and start the development server:
    ```sh
    npm install
    npm run dev
    ```
- The app runs at `http://localhost:3000`

---

Thanks for checking out my work. This repository is my attempt to gain hands-on experience with LangChain and modern AI tooling to build full-stack agent applications from design to deployment using proprietary and open-source LLMs. Would appreciate any feedback and comments.
