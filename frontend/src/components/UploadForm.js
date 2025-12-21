import React, { useState } from 'react';

function UploadForm({ onSuccess, apiUrl }) {
  const [taskTopic, setTaskTopic] = useState('');
  const [audioFile, setAudioFile] = useState(null);
  const [dialect, setDialect] = useState('US');
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  const allowedFormats = ['.ogg', '.opus', '.mp3', '.m4a', '.wav', '.webm'];

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const fileName = file.name.toLowerCase();
    const hasValidExtension = allowedFormats.some(ext => fileName.endsWith(ext));

    if (!hasValidExtension) {
      setUploadError(`Invalid format. Allowed: ${allowedFormats.join(', ')}`);
      setAudioFile(null);
      return;
    }

    setUploadError(null);
    setAudioFile(file);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!taskTopic.trim() || !audioFile) {
      setUploadError('Please fill in all fields');
      return;
    }

    if (taskTopic.trim().length < 3) {
      setUploadError('Task topic must be at least 3 characters');
      return;
    }

    setUploading(true);
    setUploadError(null);

    try {
      const formData = new FormData();
      formData.append('audio_file', audioFile);
      formData.append('task_topic', taskTopic.trim());
      formData.append('dialect', dialect);

      const response = await fetch(`${apiUrl}/api/submission`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Upload failed');
      }

      const data = await response.json();
      onSuccess(data.submission_id, taskTopic.trim());
      setTaskTopic('');
      setAudioFile(null);
    } catch (err) {
      setUploadError(`Upload failed: ${err.message}`);
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <form className="upload-form" onSubmit={handleSubmit}>
      {uploadError && <div className="error-text">{uploadError}</div>}

      <div className="form-group">
        <label htmlFor="task-topic">📋 Task Topic / Prompt</label>
        <input
          id="task-topic"
          type="text"
          placeholder="e.g., Describe your favorite hobby"
          value={taskTopic}
          onChange={(e) => setTaskTopic(e.target.value)}
          disabled={uploading}
          required
        />
      </div>

      <div className="form-group">
        <label htmlFor="audio-file">🎤 Audio File (WhatsApp voice notes supported)</label>
        <input
          id="audio-file"
          type="file"
          accept={allowedFormats.join(',')}
          onChange={handleFileChange}
          disabled={uploading}
          required
        />
        <small>Supported: {allowedFormats.join(', ')}</small>
      </div>

      <div className="form-group">
        <label htmlFor="dialect">🌍 English Dialect (optional)</label>
        <select
          id="dialect"
          value={dialect}
          onChange={(e) => setDialect(e.target.value)}
          disabled={uploading}
        >
          <option value="US">US English</option>
          <option value="UK">UK English</option>
        </select>
      </div>

      {audioFile && (
        <div className="file-info">
          <strong>Selected:</strong> {audioFile.name} ({(audioFile.size / 1024).toFixed(2)} KB)
        </div>
      )}

      <button type="submit" className="btn btn-primary" disabled={uploading}>
        {uploading ? '⏳ Uploading...' : '📤 Upload Audio'}
      </button>
    </form>
  );
}

export default UploadForm;
