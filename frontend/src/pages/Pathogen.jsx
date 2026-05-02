import React, { useState } from 'react';
import { Beaker, ShieldAlert, Cpu } from 'lucide-react';

export default function Pathogen() {
  const [beta, setBeta] = useState(0.30);
  const [mu, setMu] = useState(0.01);
  const [days, setDays] = useState(100);
  const [preset, setPreset] = useState('COVID-19 (Standard)');

  const handlePreset = (e) => {
    setPreset(e.target.value);
    if(e.target.value === 'COVID-19 (Standard)') { setBeta(0.30); setMu(0.01); }
    if(e.target.value === 'COVID-19 (Delta)') { setBeta(0.50); setMu(0.02); }
    if(e.target.value === 'Measles (High Spread)') { setBeta(0.80); setMu(0.005); }
    if(e.target.value === 'Ebola (High Lethality)') { setBeta(0.20); setMu(0.15); }
  };

  return (
    <div className="fade-in" style={{ maxWidth: '1000px', margin: '0 auto' }}>
      
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '40px' }}>
        <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(16, 185, 129, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
          <Beaker size={24} color="var(--accent-green)" />
        </div>
        <div>
          <h1 className="text-gradient" style={{ fontSize: '36px', letterSpacing: '-1px' }}>Pathogen Logic Center</h1>
          <p style={{ color: 'var(--text-muted)' }}>Fine-tune the mathematical parameters of the viral contagion.</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
        
        {/* Left Side Controls */}
        <div className="glass-panel" style={{ padding: '32px' }}>
          <h2 style={{ fontSize: '18px', marginBottom: '16px', color: '#fff', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cpu size={20} color="var(--accent-blue)"/> Preset Configurations
          </h2>
          <select 
            value={preset} 
            onChange={handlePreset}
            style={{ 
              width: '100%', padding: '16px', borderRadius: '12px', background: 'rgba(0,0,0,0.3)', 
              border: '1px solid var(--glass-border)', color: '#fff', marginBottom: '32px',
              fontFamily: 'inherit', fontSize: '16px', outline: 'none', cursor: 'pointer'
            }}
          >
            <option>COVID-19 (Standard)</option>
            <option>COVID-19 (Delta)</option>
            <option>Measles (High Spread)</option>
            <option>Ebola (High Lethality)</option>
          </select>

          <div style={{ height: '1px', background: 'var(--glass-border)', margin: '32px 0' }} />

          <h2 style={{ fontSize: '18px', marginBottom: '24px', color: '#fff', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ShieldAlert size={20} color="var(--accent-red)"/> Sub-Atomic Overrides
          </h2>
          
          <SliderControl label="Transmission Factor (Beta)" value={beta} min={0} max={1} step={0.01} onChange={(e) => setBeta(parseFloat(e.target.value))} color="var(--accent-yellow)" />
          <SliderControl label="Base Mortality Factor (Mu)" value={mu} min={0} max={0.5} step={0.001} onChange={(e) => setMu(parseFloat(e.target.value))} color="var(--accent-red)" />
          <SliderControl label="Epoch Horizon (Days)" value={days} min={30} max={365} step={1} onChange={(e) => setDays(parseFloat(e.target.value))} isInt color="var(--accent-blue)" />

        </div>

        {/* Right Side Info Box */}
        <div className="glass-panel glass-panel-glow" style={{ padding: '32px', alignSelf: 'start', position: 'relative', overflow: 'hidden' }}>
          
          <div style={{ position: 'absolute', top: '-50px', right: '-50px', width: '150px', height: '150px', background: 'var(--accent-green)', filter: 'blur(80px)', opacity: '0.2', zIndex: 0 }} />
          
          <h4 style={{ color: 'var(--accent-green)', marginBottom: '24px', fontSize: '20px', position: 'relative', zIndex: 1 }}>
            SEAIRD Integration Matrix
          </h4>
          <p style={{ color: 'var(--text-muted)', lineHeight: '1.7', fontSize: '15px', marginBottom: '24px', position: 'relative', zIndex: 1 }}>
            Constants dynamically piped into differential matrices:
          </p>
          <div style={{ background: 'rgba(0,0,0,0.4)', borderRadius: '12px', padding: '16px', marginBottom: '24px', border: '1px solid var(--glass-border)', position: 'relative', zIndex: 1 }}>
             <p style={{ color: '#fff', marginBottom: '12px', display: 'flex', justifyContent: 'space-between' }}>
               <span>Susceptible → Exposed:</span>
               <strong style={{color:'var(--accent-yellow)'}}>β = {beta.toFixed(2)}</strong>
             </p>
             <p style={{ color: '#fff', display: 'flex', justifyContent: 'space-between' }}>
               <span>Infected → Deceased:</span>
               <strong style={{color:'var(--accent-red)'}}>μ = {mu.toFixed(3)}</strong>
             </p>
          </div>
          
          <div style={{ padding: '16px', borderLeft: '3px solid var(--accent-blue)', background: 'rgba(56, 189, 248, 0.05)', position: 'relative', zIndex: 1  }}>
            <p style={{ color: '#fff', fontStyle: 'italic', fontSize: '14px', lineHeight: '1.6' }}>
              "The AI agents do not perceive these values initially. Discovery via episodic reinforcement iteration is required."
            </p>
          </div>
        </div>
        
      </div>
      <style>{`
        input[type=range] {
          -webkit-appearance: none; width: 100%; background: transparent;
        }
        input[type=range]::-webkit-slider-thumb {
          -webkit-appearance: none; height: 20px; width: 20px; border-radius: 50%;
          background: #ffffff; cursor: pointer; margin-top: -8px; box-shadow: 0 0 10px rgba(255,255,255,0.5);
        }
        input[type=range]::-webkit-slider-runnable-track {
          width: 100%; height: 4px; cursor: pointer; background: rgba(255,255,255,0.1); border-radius: 2px;
        }
      `}</style>
    </div>
  );
}

function SliderControl({ label, value, min, max, step, onChange, isInt, color }) {
  const percent = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ marginBottom: '32px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
        <span style={{ fontSize: '15px', color: '#fff' }}>{label}</span>
        <span style={{ color: color, fontSize: '16px', fontWeight: 'bold' }}>{isInt ? value : value.toFixed(strToDp(step))}</span>
      </div>
      <div style={{ position: 'relative', paddingTop: '8px' }}>
        <div style={{ position: 'absolute', top: '10px', left: 0, height: '4px', background: color, width: `${percent}%`, borderRadius: '2px', pointerEvents: 'none', boxShadow: `0 0 10px ${color}` }} />
        <input 
          type="range" 
          min={min} max={max} step={step} 
          value={value} 
          onChange={onChange}
        />
      </div>
    </div>
  )
}

function strToDp(num) {
  if(num % 1 !== 0) return num.toString().split('.')[1].length;
  return 0;
}
