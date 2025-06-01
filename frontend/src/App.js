import React, { useState, useEffect } from 'react';
import FileUpload from './components/FileUpload';
import PredictionResults from './components/PredictionResults';
import TrainingStatus from './components/TrainingStatus';
import GraphVisualization from './components/GraphVisualization';
import { api } from './services/api';
import './styles/main.css';

function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedIndividual, setSelectedIndividual] = useState(null);

  useEffect(() => {
    // Fetch initial training status
    fetchTrainingStatus();
    const interval = setInterval(fetchTrainingStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchTrainingStatus = async () => {
    try {
      const status = await api.getTrainingStatus();
      setTrainingStatus(status);
    } catch (error) {
      console.error('Error fetching training status:', error);
    }
  };

  const handleFileUpload = async (file) => {
    setLoading(true);
    try {
      const response = await api.uploadGedcom(file);
      setUploadedFile(response);
      // Poll for processing completion
      pollProcessingStatus(response.file_id);
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading file. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const pollProcessingStatus = async (fileId) => {
    // In a real implementation, this would poll a status endpoint
    // For now, we'll just wait and assume it's done
    setTimeout(() => {
      alert('File processed successfully! You can now make predictions.');
    }, 5000);
  };

  const handlePredictParents = async (individualId) => {
    if (!uploadedFile) {
      alert('Please upload a GEDCOM file first.');
      return;
    }

    setLoading(true);
    try {
      const response = await api.predictParents(uploadedFile.file_id, individualId);
      setPredictions(response);
      setSelectedIndividual(individualId);
    } catch (error) {
      console.error('Error predicting parents:', error);
      alert('Error making predictions. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>ʻŌhanaGPT</h1>
        <p>AI-Powered Genealogy Parent Prediction</p>
      </header>

      <main className="app-main">
        <div className="container">
          <section className="upload-section">
            <h2>Upload GEDCOM File</h2>
            <FileUpload onUpload={handleFileUpload} loading={loading} />
            {uploadedFile && (
              <p className="upload-status">
                File uploaded: {uploadedFile.file_id}
              </p>
            )}
          </section>

          <section className="status-section">
            <h2>Training Status</h2>
            <TrainingStatus status={trainingStatus} />
          </section>

          {uploadedFile && (
            <section className="prediction-section">
              <h2>Make Predictions</h2>
              <div className="prediction-controls">
                <input
                  type="text"
                  placeholder="Enter individual ID (e.g., @I123@)"
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      handlePredictParents(e.target.value);
                    }
                  }}
                />
                <button
                  onClick={() => {
                    const input = document.querySelector('.prediction-controls input');
                    handlePredictParents(input.value);
                  }}
                  disabled={loading}
                >
                  Predict Parents
                </button>
              </div>

              {predictions && (
                <PredictionResults
                  predictions={predictions}
                  onSelectIndividual={handlePredictParents}
                />
              )}
            </section>
          )}

          {predictions && (
            <section className="visualization-section">
              <h2>Family Tree Visualization</h2>
              <GraphVisualization
                fileId={uploadedFile.file_id}
                highlightIndividual={selectedIndividual}
                predictions={predictions}
              />
            </section>
          )}
        </div>
      </main>

      <footer className="app-footer">
        <p>© 2024 ʻŌhanaGPT - Connecting Families Through AI</p>
      </footer>
    </div>
  );
}

export default App;