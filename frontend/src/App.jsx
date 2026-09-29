import { useState } from 'react';
import './index.css';

function App() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const [formData, setFormData] = useState({
    system_id: 'prod-cluster-1',
    timestamp_start: '1695970000',
    timestamp_end: '1695973600'
  });

  const handleInputChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleAnalyze = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          system_id: formData.system_id,
          timestamp_start: parseInt(formData.timestamp_start, 10),
          timestamp_end: parseInt(formData.timestamp_end, 10)
        }),
      });

      if (!response.ok) {
        throw new Error('Analysis failed. Ensure backend API is running.');
      }

      const data = await response.json();
      
      // Simulate slight network delay for better UX on fast local responses
      setTimeout(() => {
        setResults(data);
        setLoading(false);
      }, 800);
      
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  // Convert the object structure {service: [0], pod: [2]} into flat array for mapping
  const renderFaultyEntities = () => {
    if (!results || !results.faulty_entities) return null;
    
    const entities = [];
    Object.entries(results.faulty_entities).forEach(([type, ids]) => {
      ids.forEach(id => {
        entities.push({ type, id });
      });
    });

    if (entities.length === 0) {
      return (
        <div className="glass-panel entity-card animate-fade-in">
          <div className="entity-type">SYSTEM HEALTH</div>
          <div className="entity-id" style={{color: 'var(--success)'}}>Healthy</div>
          <div className="fault-details">No anomalies detected in the selected timeframe.</div>
        </div>
      );
    }

    return (
      <div className="entities-grid animate-fade-in delay-1">
        {entities.map((entity, idx) => (
          <div key={`${entity.type}-${entity.id}-${idx}`} className="glass-panel entity-card">
            <div className="entity-type">{entity.type}</div>
            <div className="entity-id">{`${entity.type.toUpperCase()}-${entity.id}`}</div>
            <div className="fault-details">Anomaly Detected</div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <>
      <header className="app-header glass-panel" style={{borderRadius: 0, borderTop: 'none', borderLeft: 'none', borderRight: 'none'}}>
        <div className="logo">HolisticRCA.</div>
        <div className="status-badge status-success">System Online</div>
      </header>

      <main className="dashboard-container">
        {/* Left Sidebar: Analysis Form */}
        <aside>
          <div className="glass-panel analysis-form-container">
            <div>
              <h2 className="form-title">Analysis Parameters</h2>
              <p style={{color: 'var(--text-muted)', fontSize: '14px', marginBottom: '16px'}}>Specify the time window and system target to run deep observability analysis.</p>
            </div>
            
            <form onSubmit={handleAnalyze} style={{display: 'flex', flexDirection: 'column', gap: '16px'}}>
              <div className="form-group">
                <label className="form-label">System ID</label>
                <input 
                  type="text" 
                  name="system_id"
                  className="form-input" 
                  value={formData.system_id}
                  onChange={handleInputChange}
                  required
                />
              </div>
              
              <div className="form-group">
                <label className="form-label">Start Timestamp</label>
                <input 
                  type="text" 
                  name="timestamp_start"
                  className="form-input" 
                  value={formData.timestamp_start}
                  onChange={handleInputChange}
                  required
                />
              </div>

              <div className="form-group">
                <label className="form-label">End Timestamp</label>
                <input 
                  type="text" 
                  name="timestamp_end"
                  className="form-input" 
                  value={formData.timestamp_end}
                  onChange={handleInputChange}
                  required
                />
              </div>

              <button type="submit" className="btn-primary" disabled={loading}>
                {loading ? 'Analyzing...' : 'Run Root Cause Analysis'}
              </button>
            </form>

            {error && (
              <div className="status-badge status-danger" style={{marginTop: '16px', display: 'inline-block', textAlign: 'center'}}>
                {error}
              </div>
            )}
          </div>
        </aside>

        {/* Right Area: Results Display */}
        <section className="results-container">
          {!loading && !results && !error && (
            <div className="glass-panel empty-state">
              <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" style={{marginBottom: '16px', opacity: 0.5}}>
                <path d="M2 12h4l3-9 5 18 3-9h5"/>
              </svg>
              <h2>Awaiting Analysis</h2>
              <p style={{marginTop: '8px', maxWidth: '400px'}}>Select parameters and run the analysis to view detected faults and root causes based on trace, metric, and log data.</p>
            </div>
          )}

          {loading && (
            <div className="glass-panel loader-container">
              <div className="spinner"></div>
              <h3>Analyzing Observability Data</h3>
              <p>Running mask learning & GNN fusion...</p>
            </div>
          )}

          {results && (
            <>
              <div>
                <div className="results-header animate-fade-in">
                  <h2 className="form-title" style={{fontSize: '24px', margin: 0}}>Identified Root Causes</h2>
                  <div className={`status-badge ${Object.keys(results.faulty_entities || {}).length > 0 ? 'status-danger' : 'status-success'}`}>
                    {Object.keys(results.faulty_entities || {}).length > 0 ? 'Faults Detected' : 'Healthy'}
                  </div>
                </div>
                <p style={{color: 'var(--text-muted)', fontSize: '14px', marginBottom: '24px'}} className="animate-fade-in">
                  Based on inference across dimensions 1, 2, and 3.
                </p>
                {renderFaultyEntities()}
              </div>

              {results.recommended_actions && results.recommended_actions.length > 0 && (
                <div className="glass-panel recommendations-panel animate-fade-in delay-2">
                  <h2 className="form-title">Recommended Actions</h2>
                  <ul className="recommendations-list">
                    {results.recommended_actions.map((rec, i) => (
                      <li key={i} className="recommendation-item">
                        <span className="recommendation-icon">→</span>
                        <div>{rec}</div>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
        </section>
      </main>
    </>
  );
}

export default App;
