import React, { useState } from 'react';

function FeedbackDisplay({ data }) {
  const [activeTab, setActiveTab] = useState('teacher');

  if (!data) {
    return <p>No feedback available</p>;
  }

  const teacher = data.teacher_summary || {};
  const student = data.student_feedback || {};

  return (
    <div className="feedback-container">
      <div className="feedback-tabs">
        <button
          className={`tab-btn ${activeTab === 'teacher' ? 'active' : ''}`}
          onClick={() => setActiveTab('teacher')}
        >
          👨‍🏫 Teacher Summary
        </button>
        <button
          className={`tab-btn ${activeTab === 'student' ? 'active' : ''}`}
          onClick={() => setActiveTab('student')}
        >
          👨‍🎓 Student Feedback
        </button>
      </div>

      <div className="feedback-content">
        {activeTab === 'teacher' && (
          <div className="teacher-feedback">
            <div className="level-badge">
              Level: <strong>{data.student_level_guess || 'B1'}</strong>
            </div>

            <div className="summary-section">
              <h3>Overall Assessment</h3>
              <p>{teacher.overall || 'No assessment available'}</p>
            </div>

            {teacher.strengths && teacher.strengths.length > 0 && (
              <div className="strengths-section">
                <h3>✅ Strengths</h3>
                <ul>
                  {teacher.strengths.map((strength, idx) => (
                    <li key={idx}>{strength}</li>
                  ))}
                </ul>
              </div>
            )}

            {teacher.focus_next && teacher.focus_next.length > 0 && (
              <div className="focus-section">
                <h3>🎯 Areas for Focus</h3>
                <ul>
                  {teacher.focus_next.map((area, idx) => (
                    <li key={idx}>{area}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {activeTab === 'student' && (
          <div className="student-feedback">
            {student.quick_message && (
              <div className="quick-message">
                <h3>💬 Quick Message</h3>
                <p>{student.quick_message}</p>
              </div>
            )}

            {student.top_fixes && student.top_fixes.length > 0 && (
              <div className="corrections-section">
                <h3>📝 Suggestions</h3>
                <div className="corrections-list">
                  {student.top_fixes.map((fix, idx) => (
                    <div key={idx} className="correction-item">
                      <div className="correction-original">
                        ❌ <em>"{fix.original}"</em>
                      </div>
                      <div className="correction-better">
                        ✅ <strong>"{fix.better}"</strong>
                      </div>
                      <div className="correction-why">
                        💡 {fix.why}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {student.better_version && (
              <div className="better-version-section">
                <h3>🌟 More Natural Version</h3>
                <p className="better-text">{student.better_version}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default FeedbackDisplay;
