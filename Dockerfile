# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy all files from current directory
COPY . .

# Install required packages
RUN pip install --no-cache-dir \
    openai \
    tweepy \
    chromadb \
    python-dotenv \
    schedule \
    typing-extensions \
    dataclasses-json \
    coloredlogs \
    aiohttp \
    asyncio \
    langchain-core \
    pydantic

# Run tweet_poster.py when container launches
CMD ["python", "tweet_poster.py"]