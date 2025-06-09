# ʻŌhanaGPT - AI-Powered Genealogy Parent Prediction System

ʻŌhanaGPT is an advanced AI system that uses Graph Neural Networks and Transformer models to predict missing parent relationships in genealogical data from GEDCOM files.

## Features

- **GEDCOM File Processing**: Parse and analyze genealogical data from GEDCOM files
- **Incremental Learning**: Train models incrementally with each new upload
- **Duplicate Detection**: Intelligent deduplication to avoid redundant training
- **Multiple Model Architectures**: Support for both GNN and Transformer models
- **Dual Framework Support**: Implementation in both PyTorch and Apple's MLX
- **Real-time Predictions**: Get instant parent predictions for any individual
- **Interactive Visualization**: D3.js-powered family tree visualization
- **RESTful API**: Clean API for integration with other systems

## Technology Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **PyTorch**: Deep learning framework for GNN implementation
- **MLX**: Apple's machine learning framework for Mac optimization
- **NetworkX**: Graph processing and analysis
- **PostgreSQL**: Database for storing processed data
- **Redis**: Caching and task queue
- **Celery**: Distributed task processing

### Frontend
- **React**: Modern UI framework
- **D3.js**: Data visualization
- **Tailwind CSS**: Utility-first CSS framework

## Installation

### Prerequisites
- Docker and Docker Compose
- Python 3.10+
- Node.js 18+
- PostgreSQL 15+
- Redis 7+

### Quick Start

### Quick Start

1. Clone the repository:

git clone https://github.com/ikaikaalika/ohanagpt.git
cd ohana-gpt

Start with Docker Compose:

bashdocker-compose up -d

Access the application:


Frontend: http://localhost:3000
Backend API: http://localhost:8000
API Documentation: http://localhost:8000/docs

Manual Installation
Backend Setup
bashcd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
Frontend Setup
bashcd frontend
npm install
npm start
Usage

Upload GEDCOM File: Drag and drop your GEDCOM file or click to browse
Wait for Processing: The system will parse and train on your data
Make Predictions: Enter an individual's ID to predict potential parents
Explore Results: View confidence scores and explore family connections
Visualize: Interactive graph visualization of family relationships

API Endpoints

POST /api/v1/upload: Upload GEDCOM file
POST /api/v1/predict/{file_id}: Predict parents for an individual
GET /api/v1/training/status: Get current training status
GET /api/v1/model/info: Get model information

Model Architecture
Graph Neural Network (GNN)

Multi-layer Graph Attention Network (GAT)
Edge-type aware message passing
Skip connections for deeper networks
Attention mechanisms for relationship importance

Transformer Model

Self-attention for global relationship understanding
Positional encoding for family tree structure
Multi-head attention for diverse relationship patterns

Training Process

Data Ingestion: Parse GEDCOM file and extract individuals/families
Graph Construction: Build directed graph with typed edges
Feature Extraction: Convert individual attributes to numerical features
Incremental Training: Update model with new data while preserving previous knowledge
Deduplication: Identify and filter duplicate individuals
Model Update: Fine-tune model weights with new examples

Performance Optimization

Batch Processing: Efficient handling of large GEDCOM files
GPU Acceleration: CUDA support for PyTorch models
Metal Performance Shaders: Optimized for Apple Silicon with MLX
Caching: Redis caching for frequently accessed data
Async Processing: Non-blocking file uploads and training

Contributing
We welcome contributions! Please see our Contributing Guidelines for details.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

The genealogy community for GEDCOM standardization
PyTorch and MLX teams for excellent deep learning frameworks
All contributors and testers

Contact
For questions or support, please open an issue on GitHub or contact us at support@ohanagpt.com
