import React from 'react';

const TrainingStatus = ({ status }) => {
  if (!status) {
    return <div className="training-status loading">Loading status...</div>;
  }

  const formatDate = (dateString) => {
    if (!dateString) return 'Never';
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  return (
    <div className="training-status">
      <div className="status-grid">
        <div className="status-item">
          <h4>Total Individuals</h4>
          <p className="status-value">{status.total_individuals.toLocaleString()}</p>
        </div>
        <div className="status-item">
          <h4>Last Training</h4>
          <p className="status-value">{formatDate(status.last_training)}</p>
        </div>
        {status.last_metrics && (
          <div className="status-item">
            <h4>Last Training Loss</h4>
            <p className="status-value">
              {status.last_metrics.final_loss?.toFixed(4) || 'N/A'}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default TrainingStatus;