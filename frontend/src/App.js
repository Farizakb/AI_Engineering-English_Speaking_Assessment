import React, { useState } from 'react';
import './App.css';
import UploadForm from './components/UploadForm';
import TranscriptDisplay from './components/TranscriptDisplay';
import FeedbackDisplay from './components/FeedbackDisplay';
import ErrorAlert from './components/ErrorAlert';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [submissionId, setSubmissionId] = useState(null);
  const [taskTopic, setTaskTopic] = useState('');
  const [transcript, setTranscript] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleUploadSuccess = async (id, topic) => {
    setSubmissionId(id);
    setTaskTopic(topic);
    setTranscript(null);
    setFeedback(null);
    setError(null);
  };

  const handleTranscribe = async () => {
    if (!submissionId) {
      setError('No submission selected');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/submission/${submissionId}/transcribe`,
        { method: 'POST' }
      );

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Transcription failed');
      }

      const data = await response.json();
      setTranscript({
        text: data.transcript,
        duration: data.duration_seconds,
        wordCount: data.word_count,
        confidence: data.confidence
      });
    } catch (err) {
      setError(`Transcription error: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateFeedback = async () => {
    if (!submissionId || !transcript) {
      setError('Transcript not available');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/submission/${submissionId}/feedback`,
        { method: 'POST' }
      );

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Feedback generation failed');
      }

      const data = await response.json();
      setFeedback(data.feedback);
    } catch (err) {
      setError(`Feedback error: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSubmissionId(null);
    setTaskTopic('');
    setTranscript(null);
    setFeedback(null);
    setError(null);
    setUploadProgress(0);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>📚 English Speaking Assessment</h1>
        <p>Transcribe and evaluate student speaking submissions</p>
      </header>

      <main className="app-main">
        {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}

        <div className="workflow-container">
          {/* Step 1: Upload */}
          <section className="workflow-step">
            <h2>Step 1: Upload Audio</h2>
            {!submissionId ? (
              <UploadForm onSuccess={handleUploadSuccess} apiUrl={API_BASE_URL} />
            ) : (
              <div className="success-box">
                <p>✓ Audio uploaded successfully</p>
                <p><strong>Submission ID:</strong> {submissionId}</p>
                <p><strong>Task:</strong> {taskTopic}</p>
              </div>
            )}
          </section>

          {/* Step 2: Transcribe */}
          {submissionId && (
            <section className="workflow-step">
              <h2>Step 2: Transcribe Audio</h2>
              {transcript ? (
                <TranscriptDisplay data={transcript} />
              ) : (
                <button
                  className="btn btn-primary"
                  onClick={handleTranscribe}
                  disabled={loading}
                >
                  {loading ? '⏳ Transcribing...' : '🎯 Transcribe Now'}
                </button>
              )}
            </section>
          )}

          {/* Step 3: Generate Feedback */}
          {transcript && (
            <section className="workflow-step">
              <h2>Step 3: Generate Feedback</h2>
              {feedback ? (
                <FeedbackDisplay data={feedback} />
              ) : (
                <button
                  className="btn btn-secondary"
                  onClick={handleGenerateFeedback}
                  disabled={loading}
                >
                  {loading ? '⏳ Generating...' : '💡 Generate Feedback'}
                </button>
              )}
            </section>
          )}

          {/* Reset Button */}
          {submissionId && (
            <section className="workflow-step">
              <button className="btn btn-reset" onClick={handleReset}>
                🔄 Start Over
              </button>
            </section>
          )}
        </div>
      </main>

      <footer className="app-footer">
        <p>MVP • English Speaking Assessment System</p>
        <p style={{ fontSize: '0.9em', marginTop: '0.5rem' }}>
          Powered by Faster-Whisper STT and OpenAI LLM
        </p>
      </footer>
    </div>
  );
}

export default App;
