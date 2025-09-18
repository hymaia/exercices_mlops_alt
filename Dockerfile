FROM python:3.12-slim


# Install Poetry
RUN pip install poetry

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-root

# Copy project files
COPY . .
# Expose port
EXPOSE 5050
# Serve the model
CMD ["poetry", "run", "mlflow", "models", "serve", "-m", "mlops_exo/mlruns/526187138122875253/a81df006786f4849815aa85f0d80baaf/artifacts/model", "-h", "0.0.0.0", "-p", "5050", "--no-conda"]
