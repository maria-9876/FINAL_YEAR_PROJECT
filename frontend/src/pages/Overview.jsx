import React from 'react';
import { Target, Shield, Coins, Share2 } from 'lucide-react';

export default function Overview() {
  return (
    <div className="fade-in" style={{ maxWidth: '1000px', margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '12px' }}>
        <span style={{ fontSize: '40px', filter: 'drop-shadow(0 0 10px rgba(56,189,248,0.5))' }}>🌍</span>
        <h1 className="text-gradient-primary" style={{ fontSize: '42px', letterSpacing: '-1px' }}>
          Kerala AI Endemic Allocator
        </h1>
      </div>
      <p style={{ fontSize: '18px', color: 'var(--text-muted)', marginBottom: '48px', maxWidth: '600px', lineHeight: '1.6' }}>
        Welcome to the strictly scientific, data-driven resource allocator modeling viral outbreaks in Kerala via Multi-Agent Reinforcement Learning.
      </p>

      <h2 style={{ fontSize: '24px', marginBottom: '24px', color: '#fff' }}>Optimization Vectors</h2>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', marginBottom: '40px' }}>
        <VectorCard icon={<Shield size={28} color="var(--accent-red)" />} title="Infection Containment" desc="Automated dynamic lockdown mandates across isolated zones." />
        <VectorCard icon={<Target size={28} color="var(--accent-blue)" />} title="Hospital Integrity" desc="Load-balancing Ventilators and ICU beds to prevent state collapse." />
        <VectorCard icon={<Coins size={28} color="var(--accent-yellow)" />} title="Economic Growth" desc="Minimizing GDP damage and preventing total bankruptcy." />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: '24px' }}>
        <div className="glass-panel" style={{ padding: '32px' }}>
          <h3 style={{ fontSize: '20px', marginBottom: '16px', color: '#fff' }}>Operational Pipeline</h3>
          <ol style={{ paddingLeft: '24px', lineHeight: '2.5', color: 'var(--text-muted)' }}>
            <li><strong style={{color:'#fff'}}>Demographics Grid:</strong> Alter base population behaviors.</li>
            <li><strong style={{color:'#fff'}}>Pathogen Logic:</strong> Define transmission (Beta) and mortality (Mu).</li>
            <li><strong style={{color:'#fff'}}>Neural Execution:</strong> Initialize simulation and wait for agent convergence.</li>
          </ol>
        </div>

        <div className="glass-panel glass-panel-glow" style={{ padding: '32px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
            <Share2 size={24} color="var(--accent-blue)" />
            <h4 style={{ color: '#fff', fontSize: '18px' }}>CTDE Architecture</h4>
          </div>
          <p style={{ color: 'var(--text-muted)', lineHeight: '1.7', fontSize: '15px' }}>
            The 14 Reinforcement Learning Agents leverage the <strong style={{color: 'var(--accent-blue)'}}>Decentralized Actor-Centralized Critic</strong> paradigm. Agents perceive entirely local spaces but are mathematically evaluated simultaneously state-wide.
          </p>
        </div>
      </div>
    </div>
  );
}

function VectorCard({icon, title, desc}) {
  return (
    <div className="glass-panel" style={{ padding: '24px', transition: 'transform 0.2s', cursor: 'default' }}>
      <div style={{ marginBottom: '16px', padding: '12px', background: 'rgba(255,255,255,0.03)', display: 'inline-block', borderRadius: '12px' }}>
        {icon}
      </div>
      <h4 style={{ color: '#fff', fontSize: '18px', marginBottom: '8px' }}>{title}</h4>
      <p style={{ color: 'var(--text-muted)', fontSize: '14px', lineHeight: '1.5' }}>{desc}</p>
    </div>
  )
}
