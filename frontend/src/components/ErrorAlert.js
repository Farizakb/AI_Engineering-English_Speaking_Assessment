import React from 'react';

function ErrorAlert({ message, onDismiss }) {
  return (
    <div className="error-alert">
      <span>⚠️ {message}</span>
      <button onClick={onDismiss} className="dismiss-btn">×</button>
    </div>
  );
}

export default ErrorAlert;
