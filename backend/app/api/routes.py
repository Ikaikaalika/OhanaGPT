from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
import aiofiles
import uuid
from pathlib import Path
import logging

from ..models.gedcom_parser import GedcomParser
from ..models.graph_builder import GenealogyGraph
from ..training.trainer import IncrementalTrainer
from ..config import config

logger = logging.getLogger(__name__)
router = APIRouter()

# Global trainer instance
trainer = None

@router.on_event("startup")
async def startup_event():
    """Initialize trainer on startup."""
    global trainer
    trainer = IncrementalTrainer(
        model_type='gnn',
        framework='pytorch',
        config=config.__dict__
    )
    trainer.load_checkpoint()

@router.post("/upload")
async def upload_gedcom(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
) -> Dict:
    """Upload and process GEDCOM file."""
    # Validate file
    if not file.filename.endswith(('.ged', '.gedcom')):
        raise HTTPException(400, "Invalid file type. Please upload a GEDCOM file.")
    
    if file.size > config.MAX_UPLOAD_SIZE:
        raise HTTPException(400, f"File too large. Maximum size is {config.MAX_UPLOAD_SIZE} bytes.")
    
    # Save file
    file_id = str(uuid.uuid4())
    file_path = config.UPLOAD_DIR / f"{file_id}.ged"
    
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Parse GEDCOM in background
    background_tasks.add_task(process_gedcom, file_id, file_path)
    
    return {
        "file_id": file_id,
        "status": "processing",
        "message": "File uploaded successfully. Processing in background."
    }

async def process_gedcom(file_id: str, file_path: Path):
    """Process GEDCOM file and train model."""
    try:
        # Parse GEDCOM
        parser = GedcomParser()
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        individuals, families = parser.parse(content)
        
        # Build graph
        graph = GenealogyGraph(individuals, families)
        graph.build_graph()
        
        # Convert to training format
        x, edge_index, edge_types = graph.to_pytorch_geometric()
        
        graph_data = {
            'individuals': individuals,
            'families': families,
            'pytorch_data': (x, edge_index, edge_types),
            'node_to_idx': graph.node_to_idx,
            'idx_to_node': graph.idx_to_node
        }
        
        # Train incrementally
        global trainer
        result = trainer.train_on_new_data(graph_data, epochs=10)
        
        # Store result
        # In production, this would be stored in a database
        logger.info(f"Training completed for file {file_id}: {result}")
        
    except Exception as e:
        logger.error(f"Error processing file {file_id}: {e}")
        raise

@router.post("/predict/{file_id}")
async def predict_parents(
    file_id: str,
    individual_id: str
) -> Dict:
    """Predict potential parents for an individual."""
    # Load graph data for the file
    # In production, this would be retrieved from a database
    
    # For now, we'll parse the file again
    file_path = config.UPLOAD_DIR / f"{file_id}.ged"
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    
    # Parse GEDCOM
    parser = GedcomParser()
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    individuals, families = parser.parse(content)
    
    # Build graph
    graph = GenealogyGraph(individuals, families)
    graph.build_graph()
    
    # Convert to format for prediction
    x, edge_index, edge_types = graph.to_pytorch_geometric()
    
    graph_data = {
        'individuals': individuals,
        'families': families,
        'pytorch_data': (x, edge_index, edge_types),
        'node_to_idx': graph.node_to_idx,
        'idx_to_node': graph.idx_to_node
    }
    
    # Get predictions
    global trainer
    predictions = trainer.predict_parents(graph_data, individual_id)
    
    # Format results
    results = []
    for parent_id, score in predictions:
        parent = individuals.get(parent_id)
        if parent:
            results.append({
                'id': parent_id,
                'name': parent.name,
                'score': float(score),
                'birth_year': parent.birth_date.year if parent.birth_date else None,
                'gender': parent.gender
            })
    
    return {
        'individual_id': individual_id,
        'predictions': results
    }

@router.get("/training/status")
async def get_training_status() -> Dict:
    """Get current training status."""
    global trainer
    
    if trainer and trainer.training_history:
        latest = trainer.training_history[-1]
        return {
            'total_individuals': len(trainer.seen_individuals),
            'last_training': latest['timestamp'].isoformat(),
            'last_metrics': latest['metrics']
        }
    
    return {
        'total_individuals': 0,
        'last_training': None,
        'last_metrics': None
    }

@router.get("/model/info")
async def get_model_info() -> Dict:
    """Get model information."""
    global trainer
    
    return {
        'model_type': trainer.model_type if trainer else None,
        'framework': trainer.framework if trainer else None,
        'config': trainer.config if trainer else None
    }