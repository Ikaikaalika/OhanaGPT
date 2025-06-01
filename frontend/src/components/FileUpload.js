import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

const FileUpload = ({ onUpload, loading }) => {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onUpload(acceptedFiles[0]);
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/plain': ['.ged', '.gedcom']
    },
    maxSize: 100 * 1024 * 1024, // 100MB
    multiple: false
  });

  return (
    <div
      {...getRootProps()}
      className={`file-upload ${isDragActive ? 'drag-active' : ''} ${loading ? 'loading' : ''}`}
    >
      <input {...getInputProps()} />
      {loading ? (
        <p>Processing file...</p>
      ) : isDragActive ? (
        <p>Drop the GEDCOM file here...</p>
      ) : (
        <div>
          <p>Drag & drop a GEDCOM file here, or click to select</p>
          <p className="file-info">Supported formats: .ged, .gedcom (max 100MB)</p>
        </div>
      )}
    </div>
  );
};

export default FileUpload;