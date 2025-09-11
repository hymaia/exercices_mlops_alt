FROM python:3.12-slim

WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install

# No additional dependencies needed

# Copy project files
COPY . .

# Expose port
EXPOSE 5050

# Serve the model
CMD ["poetry", "run", "mlflow", "models", "serve", "-m", "mlruns/920742605743189808/8dc293aa0fb84b41a1917c71ac0dfc62/artifacts/model", "-h", "0.0.0.0", "-p", "5050", "--no-conda"]
