import React from 'react';

const PredictionResults = ({ predictions, onSelectIndividual }) => {
  if (!predictions || !predictions.predictions) {
    return null;
  }

  return (
    <div className="prediction-results">
      <h3>Predicted Parents for {predictions.individual_id}</h3>
      <div className="results-grid">
        {predictions.predictions.map((prediction, index) => (
          <div key={prediction.id} className="prediction-card">
            <div className="prediction-rank">#{index + 1}</div>
            <div className="prediction-info">
              <h4>{prediction.name || 'Unknown'}</h4>
              <p className="prediction-id">{prediction.id}</p>
              <div className="prediction-details">
                <span>Birth: {prediction.birth_year || 'Unknown'}</span>
                <span>Gender: {prediction.gender || 'Unknown'}</span>
              </div>
              <div className="prediction-score">
                <div className="score-bar">
                  <div 
                    className="score-fill" 
                    style={{ width: `${prediction.score * 100}%` }}
                  />
                </div>
                <span className="score-text">
                  {(prediction.score * 100).toFixed(1)}% confidence
                </span>
              </div>
              <button
                className="explore-btn"
                onClick={() => onSelectIndividual(prediction.id)}
              >
                Explore Family
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PredictionResults;