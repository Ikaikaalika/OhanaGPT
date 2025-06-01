const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

export const api = {
  async uploadGedcom(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }

    return response.json();
  },

  async predictParents(fileId, individualId) {
    const response = await fetch(`${API_BASE_URL}/predict/${fileId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      # ʻŌhanaGPT - AI-Powered Genealogy Parent Prediction System

## Project Structure