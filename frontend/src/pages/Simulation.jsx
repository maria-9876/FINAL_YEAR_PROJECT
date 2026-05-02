import React, { useState } from 'react';
import { outbreakCurveData as mockOutbreakCurveData, finalDirectives as mockFinalDirectives, logisticsData as mockLogisticsData } from '../data/mockData';
import { LineChart, Line, XAxis, YAxis, Tooltip as RTC, ResponsiveContainer, Area, AreaChart, CartesianGrid, ScatterChart, Scatter, ZAxis, Cell, LabelList } from 'recharts';
import { Activity, Radio, AlertTriangle, Loader2 } from 'lucide-react';

export default function Simulation() {
  const [ran, setRan] = useState(false);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [outbreakCurveData, setOutbreakCurveData] = useState([]);
  const [finalDirectives, setFinalDirectives] = useState([]);
  const [logisticsData, setLogisticsData] = useState([]);

  const executeSim = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/simulate');
      if (!response.ok) throw new Error('Simulation failed');
      const data = await response.json();
      setOutbreakCurveData(data.outbreakCurveData);
      setFinalDirectives(data.finalDirectives);
      setLogisticsData(data.logisticsData);
    } catch (err) {
      console.error(err);
      setOutbreakCurveData(mockOutbreakCurveData);
      setFinalDirectives(mockFinalDirectives);
      setLogisticsData(mockLogisticsData);
    } finally {
      setLoading(false);
      setRan(true);
    }
  };

  return (
    <div className="fade-in" style={{ maxWidth: '1200px', margin: '0 auto', paddingBottom: '60px' }}>
      
      {!ran ? (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '60vh', textAlign: 'center' }}>
          <div style={{ width: '80px', height: '80px', borderRadius: '50%', background: 'rgba(244, 63, 94, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '24px', animation: 'pulseBg 2s infinite' }}>
            <Activity size={40} color="var(--accent-red)" />
          </div>
          <h1 className="text-gradient" style={{ fontSize: '48px', marginBottom: '16px', letterSpacing: '-1px' }}>Neural Execution Node</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '18px', maxWidth: '500px', marginBottom: '40px' }}>
            Sequence initiated. Awaiting command to pass pathogen rules and district tensors into the distributed RL matrix.
          </p>
          <button className="btn-primary" onClick={executeSim} disabled={loading} style={{ padding: '20px 40px', fontSize: '18px', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: '12px', opacity: loading ? 0.7 : 1 }}>
            {loading ? <><Loader2 className="animate-spin" size={24} /> Executing...</> : "Execute Multi-Agent Simulation"}
          </button>
        </div>
      ) : (
        <div className="fade-in">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '40px' }}>
            <h1 className="text-gradient" style={{ fontSize: '36px', letterSpacing: '-1px' }}>Simulation Results</h1>
            <div style={{ padding: '12px 20px', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '12px', color: 'var(--accent-green)', display: 'flex', alignItems: 'center', gap: '8px', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
              <Radio size={18} className="animate-pulse" /> Sequence Complete
            </div>
          </div>

          <div className="glass-panel" style={{ display: 'flex', padding: '8px', marginBottom: '32px', gap: '8px' }}>
            <TabLink active={activeTab === 'overview'} onClick={() => setActiveTab('overview')} icon="📊">Diagnostics</TabLink>
            <TabLink active={activeTab === 'geo'} onClick={() => setActiveTab('geo')} icon="🗺️">Geospatial Overlay</TabLink>
            <TabLink active={activeTab === 'logistics'} onClick={() => setActiveTab('logistics')} icon="🚑">Logistics Array</TabLink>
          </div>

          {activeTab === 'overview' && (
            <div className="fade-in">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px', marginBottom: '32px' }}>
                <MetricBox title="Total Active Cases" val="738,825" color="var(--accent-red)" glow="rgba(244, 63, 94, 0.2)" />
                <MetricBox title="ICU Resources Engaged" val="5,347" color="var(--accent-blue)" glow="rgba(56, 189, 248, 0.2)" />
                <MetricBox title="State Economic Retention" val="77.7%" color="var(--accent-yellow)" glow="rgba(251, 191, 36, 0.2)" />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '24px' }}>
                <div className="glass-panel" style={{ padding: '32px', position: 'relative', overflow: 'hidden' }}>
                  <h3 style={{ fontSize: '18px', color: '#fff', marginBottom: '24px', letterSpacing: '0.5px' }}>Outbreak Curve Projection</h3>
                  <div style={{ height: '350px', width: '100%', position: 'relative', zIndex: 1 }}>
                    <ResponsiveContainer>
                      <AreaChart data={outbreakCurveData} margin={{top: 10, right: 10, left: 0, bottom: 0}}>
                        <defs>
                          <linearGradient id="colorCases" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="var(--accent-red)" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="var(--accent-red)" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="day" stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} />
                        <YAxis stroke="var(--text-muted)" fontSize={12} tickFormatter={t => `${t/1000}k`} tickLine={false} axisLine={false} />
                        <RTC contentStyle={{ backgroundColor: 'rgba(3,7,18,0.9)', borderColor: 'var(--accent-red)', color: '#fff', borderRadius: '8px' }} itemStyle={{color: 'var(--accent-red)'}} />
                        <Area type="monotone" dataKey="cases" stroke="var(--accent-red)" strokeWidth={3} fillOpacity={1} fill="url(#colorCases)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="glass-panel" style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                  <div style={{ padding: '24px', borderBottom: '1px solid var(--glass-border)', background: 'rgba(255,255,255,0.02)' }}>
                    <h3 style={{ fontSize: '18px', color: '#fff', letterSpacing: '0.5px' }}>Agent Directives</h3>
                  </div>
                  <div style={{ height: '350px', overflowY: 'auto', padding: '16px' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '14px' }}>
                      <tbody>
                        {finalDirectives.map((row, i) => (
                          <tr key={row.district} style={{ borderBottom: i === finalDirectives.length-1 ? 'none' : '1px solid rgba(255,255,255,0.03)' }}>
                            <td style={{ padding: '16px 8px', color: '#fff' }}>{row.district}</td>
                            <td style={{ padding: '16px 8px' }}>
                              <span style={{ padding: '4px 8px', borderRadius: '4px', background: row.policy.includes('3') ? 'rgba(244,63,94,0.1)' : 'rgba(56,189,248,0.1)', color: row.policy.includes('3') ? 'var(--accent-red)' : 'var(--accent-blue)', fontSize: '12px', fontWeight: 'bold' }}>
                                {row.policy}
                              </span>
                            </td>
                            <td style={{ padding: '16px 8px', textAlign: 'right', color: 'var(--text-muted)' }}>{row.cases.toLocaleString()}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'geo' && (
            <div className="fade-in glass-panel" style={{ height: '550px', display: 'flex', flexDirection: 'column', padding: '24px', position: 'relative' }}>
              <div style={{ marginBottom: '16px' }}>
                <h3 style={{ fontSize: '18px', color: '#fff', letterSpacing: '0.5px' }}>Live Infection Outbreak Radar</h3>
                <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>Bubble size corresponds to active case volume. Plotted on standard Latitude/Longitude matrix without external map dependencies.</p>
              </div>
              <div style={{ flex: 1, width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis type="number" dataKey="lon" name="Longitude" domain={['dataMin - 0.2', 'dataMax + 0.2']} stroke="var(--text-muted)" tick={false} axisLine={false} />
                    <YAxis type="number" dataKey="lat" name="Latitude" domain={['dataMin - 0.2', 'dataMax + 0.2']} stroke="var(--text-muted)" tick={false} axisLine={false} />
                    <ZAxis type="number" dataKey="cases" range={[200, 3000]} name="Active Cases" />
                    <RTC cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: 'rgba(3,7,18,0.9)', borderColor: 'var(--accent-red)', color: '#fff', borderRadius: '8px' }} />
                    <Scatter name="Districts" data={finalDirectives} opacity={0.7}>
                      {
                        finalDirectives.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.policy.includes('3') ? 'var(--accent-red)' : 'var(--accent-blue)'} />
                        ))
                      }
                      <LabelList dataKey="district" position="top" fill="var(--text-muted)" fontSize={11} offset={10} />
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {activeTab === 'logistics' && (
            <div className="fade-in">
              <div className="glass-panel" style={{ overflowX: 'auto', marginBottom: '40px', padding: '1px' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '15px' }}>
                  <thead>
                    <tr style={{ background: 'rgba(255,255,255,0.02)', color: 'var(--text-muted)' }}>
                      <th style={{ padding: '20px' }}>District Node</th>
                      <th style={{ padding: '20px' }}>Bed Stress (%)</th>
                      <th style={{ padding: '20px' }}>Beds Inducted</th>
                      <th style={{ padding: '20px' }}>Hard Limit</th>
                      <th style={{ padding: '20px', color: 'var(--accent-blue)' }}>O2 Logistics</th>
                    </tr>
                  </thead>
                  <tbody>
                    {logisticsData.map((row, i) => (
                      <tr key={row.district} style={{ borderTop: '1px solid var(--glass-border)' }}>
                        <td style={{ padding: '16px 20px', color: '#fff', fontWeight: '500' }}>{row.district}</td>
                        <td style={{ padding: '16px 20px', color: row.cap > 100 ? 'var(--accent-red)' : 'var(--accent-green)', fontWeight: 'bold' }}>{row.cap.toFixed(1)}%</td>
                        <td style={{ padding: '16px 20px', color: 'var(--text-muted)' }}>{row.used}</td>
                        <td style={{ padding: '16px 20px', color: 'var(--text-muted)' }}>{row.limit}</td>
                        <td style={{ padding: '16px 20px', color: row.o2 < 0 ? 'var(--accent-red)' : 'var(--accent-blue)' }}>{row.o2}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px' }}>
                {logisticsData.filter(d => d.cap > 90).slice(0, 6).map(row => (
                  <div key={row.district} className="glass-panel" style={{ padding: '20px', borderLeft: '4px solid var(--accent-red)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                      <AlertTriangle size={18} color="var(--accent-red)" />
                      <h4 style={{ color: '#fff', fontSize: '16px' }}>{row.district}</h4>
                    </div>
                    <p style={{ color: 'var(--text-muted)', fontSize: '13px', lineHeight: '1.5' }}>
                      <span style={{ color: 'var(--accent-red)' }}>CRITICAL COLLAPSE:</span> Network overwhelmed. No donor districts.
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      )}
      <style>{`
        .animate-pulse { animation: pulseIcon 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
        @keyframes pulseIcon { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }
        .animate-spin { animation: spin 1s linear infinite; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}

function MetricBox({ title, val, color, glow }) {
  return (
    <div className="glass-panel" style={{ padding: '32px 24px', position: 'relative', overflow: 'hidden' }}>
      <div style={{ position: 'absolute', top: '-50%', left: '-20%', width: '150%', height: '150%', background: `radial-gradient(circle, ${glow} 0%, transparent 60%)`, zIndex: 0 }} />
      <div style={{ color: 'var(--text-muted)', fontSize: '14px', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '1px', position: 'relative', zIndex: 1 }}>{title}</div>
      <div style={{ color: color, fontSize: '40px', fontWeight: 'bold', letterSpacing: '-1px', position: 'relative', zIndex: 1 }}>{val}</div>
    </div>
  );
}

function TabLink({ active, onClick, children, icon }) {
  return (
    <div onClick={onClick} style={{ flex: 1, padding: '16px', cursor: 'pointer', color: active ? '#fff' : 'var(--text-muted)', background: active ? 'rgba(255,255,255,0.05)' : 'transparent', borderRadius: '8px', transition: 'all 0.2s', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', fontSize: '15px', fontWeight: active ? '600' : '500' }}>
      <span style={{fontSize: '18px'}}>{icon}</span> {children}
    </div>
  )
}
