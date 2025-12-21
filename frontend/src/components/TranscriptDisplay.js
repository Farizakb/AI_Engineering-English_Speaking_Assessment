import React from 'react';

function TranscriptDisplay({ data }) {
  if (!data || !data.text) {
    return <p>No transcript available</p>;
  }

  return (
    <div className="transcript-box">
      <div className="transcript-meta">
        <span>⏱️ Duration: {data.duration.toFixed(1)}s</span>
        <span>📝 Words: {data.wordCount}</span>
        <span>
          ✓ Confidence:{' '}
          {data.confidence === 'high' ? '🟢' : data.confidence === 'medium' ? '🟡' : '🔴'}{' '}
          {data.confidence}
        </span>
      </div>
      <div className="transcript-text">
        <p>{data.text}</p>
      </div>
    </div>
  );
}

export default TranscriptDisplay;
