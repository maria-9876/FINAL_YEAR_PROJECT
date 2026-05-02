import React, { useState } from 'react';
import { districtDemographics } from '../data/mockData';
import { BarChart, Bar, Cell, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function Demographics() {
  const [data, setData] = useState(districtDemographics);

  const handleEdit = (idx, field, value) => {
    const newData = [...data];
    // Deep clone the object to ensure React triggers re-renders and isolate changes
    newData[idx] = { ...newData[idx], [field]: value };
    setData(newData);
  };

  return (
    <div className="fade-in" style={{ maxWidth: '1200px', margin: '0 auto', paddingBottom: '60px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '32px' }}>
        <div>
          <h1 className="text-gradient" style={{ fontSize: '36px', marginBottom: '8px', letterSpacing: '-1px' }}>Demographics Grid</h1>
          <p style={{ fontSize: '16px', color: 'var(--text-muted)' }}>
            Edit starting demographics and inject cases prior to execution.
          </p>
        </div>
        <div style={{ padding: '12px 20px', background: 'rgba(56, 189, 248, 0.1)', border: '1px solid rgba(56, 189, 248, 0.4)', borderRadius: '12px', color: 'var(--accent-blue)', fontSize: '14px', fontWeight: '500', boxShadow: '0 0 20px rgba(56,189,248,0.1)' }}>
          <span style={{marginRight: '8px'}}>⚡</span> Active Model Sync: Connected
        </div>
      </div>

      <div className="glass-panel" style={{ overflowX: 'auto', marginBottom: '48px', padding: '1px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '15px' }}>
          <thead>
            <tr style={{ background: 'rgba(255,255,255,0.03)', color: 'var(--text-muted)' }}>
              <th style={{ padding: '20px 24px', fontWeight: '500', letterSpacing: '0.5px' }}>District Node</th>
              <th style={{ padding: '20px 24px', fontWeight: '500' }}>Population</th>
              <th style={{ padding: '20px 24px', fontWeight: '500' }}>Compliance</th>
              <th style={{ padding: '20px 24px', fontWeight: '500' }}>Econ Load</th>
              <th style={{ padding: '20px 24px', fontWeight: '500', color: 'var(--accent-blue)' }}>ICU Strata</th>
              <th style={{ padding: '20px 24px', fontWeight: '500', color: 'var(--accent-red)' }}>Active Seed</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, idx) => (
              <tr key={row.district} style={{ borderTop: '1px solid var(--glass-border)' }}>
                <td style={{ padding: '16px 24px', fontWeight: '600', color: '#fff' }}>{row.district}</td>
                <td style={{ padding: '16px 24px' }}><input type="number" value={row.population} onChange={(e) => handleEdit(idx, 'population', e.target.value)} style={inputStyle} /></td>
                <td style={{ padding: '16px 24px' }}><input type="number" step="0.01" value={row.compliance} onChange={(e) => handleEdit(idx, 'compliance', e.target.value)} style={inputStyle} /></td>
                <td style={{ padding: '16px 24px' }}><input type="number" step="0.01" value={row.econ} onChange={(e) => handleEdit(idx, 'econ', e.target.value)} style={inputStyle} /></td>
                <td style={{ padding: '16px 24px' }}><input type="number" value={row.icu} onChange={(e) => handleEdit(idx, 'icu', e.target.value)} style={inputStyle} className="blue-input" /></td>
                <td style={{ padding: '16px 24px' }}><input type="number" value={row.cases} onChange={(e) => handleEdit(idx, 'cases', e.target.value)} style={inputStyle} className="red-input" /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
        <div style={{ width: '8px', height: '24px', background: 'var(--accent-blue)', borderRadius: '4px' }} />
        <h2 style={{ fontSize: '22px', color: '#fff' }}>Capacity Spread Analytics</h2>
      </div>
      
      <div className="glass-panel" style={{ height: '450px', width: '100%', padding: '24px', paddingTop: '40px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 0, right: 0, left: -20, bottom: 40}}>
            <XAxis dataKey="district" angle={-45} textAnchor="end" height={80} stroke="var(--text-muted)" fontSize={13} tickLine={false} axisLine={{ stroke: 'rgba(255,255,255,0.1)' }} />
            <YAxis stroke="var(--text-muted)" fontSize={13} tickLine={false} axisLine={false} />
            <Tooltip contentStyle={{ backgroundColor: 'rgba(17, 24, 39, 0.95)', borderColor: 'var(--glass-border)', color: '#fff', borderRadius: '12px', boxShadow: '0 10px 30px rgba(0,0,0,0.5)' }} cursor={{fill: 'rgba(255,255,255,0.02)'}} />
            <Legend verticalAlign="top" height={50} iconType="circle" />
            <Bar dataKey="icu" name="Total ICU Beds" fill="var(--accent-blue)" radius={[4, 4, 0, 0]} />
            <Bar dataKey="cases" name="Starting Active Cases" fill="var(--accent-red)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

    </div>
  );
}

const inputStyle = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: '6px',
  color: 'var(--text-main)',
  boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.3)',
  width: '100%',
  fontFamily: 'inherit',
  fontSize: '15px',
  outline: 'none',
  transition: 'border 0.2s, background 0.2s',
  padding: '10px 12px',
  cursor: 'text'
};
// Use vanilla CSS inline for hover state emulation
document.head.insertAdjacentHTML("beforeend", `<style>
  input:focus { background: rgba(255,255,255,0.08) !important; border: 1px solid rgba(255,255,255,0.3) !important; }
  .blue-input:focus { border: 1px solid var(--accent-blue) !important; box-shadow: 0 0 10px rgba(56,189,248,0.2) inset !important; }
  .red-input:focus { border: 1px solid var(--accent-red) !important; box-shadow: 0 0 10px rgba(244,63,94,0.2) inset !important; }
</style>`)
