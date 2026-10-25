import { useState, useEffect } from 'react';
import './index.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [systemHealth, setSystemHealth] = useState(null);
  const [windows, setWindows] = useState([]);
  const [selectedTimestamp, setSelectedTimestamp] = useState('1696002220');
  const [epochs, setEpochs] = useState(50);
  const [targetEntity, setTargetEntity] = useState('');

  // 1. Fetch system health and available dataset observation windows on mount
  useEffect(() => {
    async function fetchInitialData() {
      try {
        const healthRes = await fetch(`${API_BASE}/api/health`);
        if (healthRes.ok) {
          const healthData = await healthRes.json();
          setSystemHealth(healthData);
        }

        const windowsRes = await fetch(`${API_BASE}/api/windows`);
        if (windowsRes.ok) {
          const windowsData = await windowsRes.json();
          const winList = windowsData.windows || [];
          setWindows(winList);

          // Find first positive validation fault window as default, else first window
          const faultWin = winList.find(w => w.has_ground_truth_fault && w.split === 'val');
          if (faultWin) {
            setSelectedTimestamp(String(faultWin.timestamp));
          } else if (winList.length > 0) {
            setSelectedTimestamp(String(winList[0].timestamp));
          }
        }
      } catch (err) {
        console.warn('Initial data fetch warning (API may not be started yet):', err);
      }
    }
    fetchInitialData();
  }, []);

  const handleAnalyze = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          timestamp: parseInt(selectedTimestamp, 10),
          epochs: parseInt(epochs, 10),
          target_entity: targetEntity.trim() ? targetEntity.trim() : null,
          run_explanation: true
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Analysis request failed. Verify backend API is running on port 8000.');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getSystemStatusBadge = () => {
    if (!systemHealth) {
      return <div className="status-badge status-warning">Checking API...</div>;
    }
    if (systemHealth.is_initialized) {
      const ep = systemHealth.checkpoint?.epoch;
      return <div className="status-badge status-success">Checkpoint Ready (Epoch {ep})</div>;
    }
    return <div className="status-badge status-danger">Checkpoint Unavailable</div>;
  };

  return (
    <>
      <header className="app-header glass-panel" style={{borderRadius: 0, borderTop: 'none', borderLeft: 'none', borderRight: 'none'}}>
        <div>
          <div className="logo">HolisticRCA.</div>
          <div style={{fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px'}}>
            Multimodal Cloud Observability & Paper-Faithful Root Cause Analysis
          </div>
        </div>
        {getSystemStatusBadge()}
      </header>

      <main className="dashboard-container">
        {/* Left Sidebar: Analysis Parameter Selection */}
        <aside>
          <div className="glass-panel analysis-form-container">
            <div>
              <h2 className="form-title">Observation Window</h2>
              <p style={{color: 'var(--text-muted)', fontSize: '13px', marginBottom: '16px'}}>
                Select from real Dataset C observation windows and execute GAT inference with relation/feature mask learning.
              </p>
            </div>

            <form onSubmit={handleAnalyze} style={{display: 'flex', flexDirection: 'column', gap: '16px'}}>
              <div className="form-group">
                <label className="form-label">Observation Window (Dataset C)</label>
                <select
                  className="form-input"
                  value={selectedTimestamp}
                  onChange={(e) => setSelectedTimestamp(e.target.value)}
                  disabled={loading}
                >
                  {windows.length === 0 ? (
                    <option value={selectedTimestamp}>Window: {selectedTimestamp}</option>
                  ) : (
                    windows.map((win) => {
                      const faultTag = win.has_ground_truth_fault
                        ? ` • FAULT: ${win.ground_truth_faults.map(f => `${f.entity} (${f.fault_type})`).join(', ')}`
                        : ' • Normal';
                      return (
                        <option key={win.timestamp} value={win.timestamp}>
                          [{win.split.toUpperCase()}] T={win.timestamp} {faultTag}
                        </option>
                      );
                    })
                  )}
                </select>
                <span style={{fontSize: '11px', color: 'var(--text-muted)'}}>
                  {windows.length} windows available across Train (36), Val (12), Test (12).
                </span>
              </div>

              <div className="form-group">
                <label className="form-label">Mask Learning Epochs (Eq. 8)</label>
                <select
                  className="form-input"
                  value={epochs}
                  onChange={(e) => setEpochs(Number(e.target.value))}
                  disabled={loading}
                >
                  <option value={25}>25 epochs (Fast Interactive)</option>
                  <option value={50}>50 epochs (Standard Convergence)</option>
                  <option value={100}>100 epochs (High Precision)</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Target Entity (Optional)</label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="Auto-detect highest fault"
                  value={targetEntity}
                  onChange={(e) => setTargetEntity(e.target.value)}
                  disabled={loading}
                />
              </div>

              <button type="submit" className="btn-primary" disabled={loading}>
                {loading ? 'Learning Masks & Explaining...' : 'Run Root Cause Analysis'}
              </button>
            </form>

            {error && (
              <div className="alert-box alert-warning animate-fade-in" style={{marginTop: '12px'}}>
                <strong>Error:</strong> {error}
              </div>
            )}

            {/* Checkpoint Diagnostics Summary */}
            {systemHealth && (
              <div style={{marginTop: '16px', borderTop: '1px solid var(--panel-border)', paddingTop: '16px', fontSize: '12px', color: 'var(--text-muted)'}}>
                <div><strong>Device:</strong> {systemHealth.device}</div>
                <div><strong>Val Loss:</strong> {systemHealth.checkpoint?.val_loss?.toFixed(4) ?? 'N/A'}</div>
                <div><strong>Entities:</strong> 14 services in RERG</div>
                <div><strong>Observability:</strong> 266 features (84T, 98M, 84L)</div>
              </div>
            )}
          </div>
        </aside>

        {/* Right Content Area: Results Display */}
        <section className="results-container">
          {/* Default Empty State */}
          {!loading && !results && !error && (
            <div className="glass-panel empty-state">
              <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{marginBottom: '16px', opacity: 0.6}}>
                <path d="M2 12h4l3-9 5 18 3-9h5"/>
              </svg>
              <h2>Select Window & Run Analysis</h2>
              <p style={{marginTop: '8px', maxWidth: '440px'}}>
                Choose an observation window from the left panel and click <strong>Run Root Cause Analysis</strong> to perform paper-faithful GAT inference, relation mask learning, and observability feature attribution.
              </p>
            </div>
          )}

          {/* Active Loading State */}
          {loading && (
            <div className="glass-panel loader-container">
              <div className="spinner"></div>
              <h3>Optimizing Masks & Performing GNN Inference</h3>
              <p style={{fontSize: '13px', color: 'var(--text-muted)'}}>
                Solving Eq. (8): Maximizing mutual information while minimizing subgraph size and discrete entropy...
              </p>
            </div>
          )}

          {/* Live RCA Results */}
          {results && !loading && (
            <div className="animate-fade-in" style={{display: 'flex', flexDirection: 'column', gap: '24px'}}>

              {/* Experimental Notice Banner */}
              <div className="alert-box alert-warning">
                <strong>EXPERIMENTAL RESEARCH DATASET:</strong> {results.experimental_notice}
              </div>

              {/* Status Header */}
              <div className="results-header">
                <div>
                  <h2 className="form-title" style={{fontSize: '22px', margin: 0}}>
                    Root Cause Analysis: Window T={results.timestamp}
                  </h2>
                  <div style={{color: 'var(--text-muted)', fontSize: '13px', marginTop: '4px'}}>
                    Split: <strong>{results.split_name?.toUpperCase()}</strong> • Evaluated on 14 Resource Entities in RERG
                  </div>
                </div>
                <div className={`status-badge ${results.is_fault_predicted ? 'status-danger' : 'status-success'}`}>
                  {results.is_fault_predicted ? 'Fault Detected' : 'Normal / Baseline'}
                </div>
              </div>

              {/* 1. Diagnostics & Prediction Summary */}
              <div className="metrics-row">
                <div className="metric-pill">
                  <div className="metric-pill-label">Max Fault Probability</div>
                  <div className="metric-pill-value" style={{color: results.is_fault_predicted ? 'var(--danger)' : 'var(--success)'}}>
                    {(results.max_fault_probability * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="metric-pill">
                  <div className="metric-pill-label">Predicted Faulty Entity</div>
                  <div className="metric-pill-value">
                    {results.top_faulty_entity || 'None (Normal)'}
                  </div>
                </div>
                <div className="metric-pill">
                  <div className="metric-pill-label">Explanation Fidelity (Prob Diff)</div>
                  <div className="metric-pill-value" style={{color: 'var(--accent-color)'}}>
                    {results.explanation?.fidelity_metrics?.prob_l1_diff !== undefined
                      ? results.explanation.fidelity_metrics.prob_l1_diff.toFixed(4)
                      : 'N/A'}
                  </div>
                </div>
                <div className="metric-pill">
                  <div className="metric-pill-label">Ground Truth Benchmark</div>
                  <div className="metric-pill-value" style={{fontSize: '14px', lineHeight: '1.4'}}>
                    {results.ground_truth_benchmark?.has_fault ? (
                      <span style={{color: '#f39c12'}}>
                        {results.ground_truth_benchmark.faults.map(f => `${f.entity}: ${f.fault_type}`).join(', ')}
                      </span>
                    ) : (
                      <span style={{color: 'var(--success)'}}>No Injected Fault (Normal)</span>
                    )}
                  </div>
                </div>
              </div>

              {/* 2. Candidate Root-Cause Entities Ranking (Dimension 1) */}
              <div className="glass-panel" style={{padding: '24px'}}>
                <h3 className="form-title" style={{fontSize: '18px', marginBottom: '4px'}}>
                  Candidate Root-Cause Entities (Dimension 1: RERG Topology & Features)
                </h3>
                <p style={{fontSize: '12px', color: 'var(--text-muted)', marginBottom: '16px'}}>
                  Entities ranked by Section IV.F learned relation mask weights (incident edge significance) and entity-specific feature mask attribution.
                </p>

                <div className="ranking-table-container">
                  <table className="ranking-table">
                    <thead>
                      <tr>
                        <th>Rank</th>
                        <th>Entity Name</th>
                        <th>RCA Score</th>
                        <th>Relation Score</th>
                        <th>Feature Score</th>
                        <th>Predicted Class</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(results.explanation?.root_cause_ranking || []).map((item, idx) => (
                        <tr key={item.entity_name} className={idx === 0 ? 'top-rank' : ''}>
                          <td>
                            <span className={`rank-badge ${idx === 0 ? 'rank-badge-1' : 'rank-badge-other'}`}>
                              #{idx + 1}
                            </span>
                          </td>
                          <td style={{fontWeight: idx === 0 ? '700' : '500'}}>
                            {item.entity_name}
                            {idx === 0 && <span style={{marginLeft: '8px', fontSize: '11px', color: 'var(--danger)'}}>(Top Root Cause)</span>}
                          </td>
                          <td style={{fontWeight: '700', color: idx === 0 ? 'var(--danger)' : 'inherit'}}>
                            {item.score?.toFixed(4)}
                          </td>
                          <td style={{color: 'var(--text-muted)'}}>{item.relation_score?.toFixed(4)}</td>
                          <td style={{color: 'var(--text-muted)'}}>{item.feature_score?.toFixed(4)}</td>
                          <td>
                            {item.predicted_fault ? (
                              <span style={{color: 'var(--danger)', fontWeight: '600'}}>{item.predicted_fault}</span>
                            ) : (
                              <span style={{color: 'var(--text-muted)'}}>normal</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* 3. Top Implicated Observability Features (Dimension 2) */}
              <div className="glass-panel" style={{padding: '24px'}}>
                <h3 className="form-title" style={{fontSize: '18px', marginBottom: '4px'}}>
                  Implicated Observability Features (Dimension 2: Traces, Metrics, Logs)
                </h3>
                <p style={{fontSize: '12px', color: 'var(--text-muted)', marginBottom: '16px'}}>
                  Top features identified by the learned observability mask σ(om). Identifies the exact telemetry signals driving the anomaly.
                </p>

                <div className="feature-bars-list">
                  {(results.explanation?.top_feature_attributions || []).slice(0, 8).map((feat, idx) => {
                    const scorePct = Math.min(100, Math.max(0, feat.score * 100));
                    return (
                      <div key={`${feat.feature_name}-${idx}`} className="feature-bar-item">
                        <div className="feature-bar-header">
                          <div>
                            <span className={`feature-modality-badge modality-${feat.modality}`}>
                              {feat.modality}
                            </span>
                            <span style={{fontWeight: '600'}}>{feat.feature_name}</span>
                            {feat.entity_name && (
                              <span style={{fontSize: '11px', color: 'var(--text-muted)', marginLeft: '8px'}}>
                                ({feat.entity_name})
                              </span>
                            )}
                          </div>
                          <div style={{fontWeight: '700', color: 'var(--accent-color)'}}>
                            {feat.score?.toFixed(4)}
                          </div>
                        </div>
                        <div className="feature-track">
                          <div className="feature-fill" style={{width: `${scorePct}%`}}></div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* 4. Optimization Diagnostics */}
              {results.explanation?.optimization_stats && (
                <div className="glass-panel" style={{padding: '20px', fontSize: '13px', color: 'var(--text-muted)'}}>
                  <div style={{fontWeight: '600', color: 'var(--text-main)', marginBottom: '8px'}}>
                    Mask Optimization Diagnostics (Eq. 8)
                  </div>
                  <div style={{display: 'flex', gap: '24px', flexWrap: 'wrap'}}>
                    <div><strong>Iterations:</strong> {results.explanation.optimization_stats.epochs} epochs</div>
                    <div><strong>Initial Loss:</strong> {results.explanation.optimization_stats.initial_loss?.toFixed(4)}</div>
                    <div><strong>Final Loss:</strong> {results.explanation.optimization_stats.final_loss?.toFixed(4)}</div>
                    <div><strong>Loss Reduction:</strong> {results.explanation.optimization_stats.loss_reduction?.toFixed(4)}</div>
                    <div><strong>Logit L1 Diff:</strong> {results.explanation.fidelity_metrics?.logit_l1_diff?.toFixed(4)}</div>
                  </div>
                </div>
              )}

            </div>
          )}
        </section>
      </main>
    </>
  );
}

export default App;
