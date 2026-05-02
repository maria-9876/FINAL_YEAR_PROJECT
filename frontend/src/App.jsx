import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { Activity, Users, Settings2, TerminalSquare } from 'lucide-react';
import Overview from './pages/Overview';
import Demographics from './pages/Demographics';
import Pathogen from './pages/Pathogen';
import Simulation from './pages/Simulation';
import './index.css';

function FloatingSidebar() {
  return (
    <nav className="glass-panel" style={{ width: 'var(--sidebar-w)', display: 'flex', flexDirection: 'column', padding: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '32px' }}>
        <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: 'linear-gradient(135deg, var(--accent-blue), var(--accent-red))', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Activity size={24} color="#fff" />
        </div>
        <h2 style={{ fontSize: '20px', fontWeight: '700', letterSpacing: '-0.5px' }}>Kerala<br/><span className="text-gradient">Command</span></h2>
      </div>
      
      <p style={{ color: 'var(--accent-blue)', fontSize: '11px', fontWeight: '600', textTransform: 'uppercase', marginBottom: '16px', letterSpacing: '1.5px', paddingLeft: '4px' }}>Control Modules</p>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <NavLink to="/" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
          <TerminalSquare size={18} color="var(--text-muted)"/> System Overview
        </NavLink>
        <NavLink to="/demographics" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
          <Users size={18} color="var(--text-muted)"/> Demographics Grid
        </NavLink>
        <NavLink to="/pathogen" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
          <Settings2 size={18} color="var(--text-muted)"/> Pathogen Logic
        </NavLink>
        <NavLink to="/simulation" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
          <Activity size={18} color="var(--text-muted)"/> Neural Execution
        </NavLink>
      </div>

      <div style={{ marginTop: 'auto', padding: '16px', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid var(--glass-border)' }}>
        <p style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.5' }}>
          Running instances: <strong style={{color: '#fff'}}>14 Agents</strong><br/>
          Architecture: <strong style={{color: 'var(--accent-blue)'}}>CTDE / PPO</strong>
        </p>
      </div>
    </nav>
  );
}

function App() {
  return (
    <Router>
      <div className="ambient-bg" />
      <FloatingSidebar />
      <main className="glass-panel" style={{ flex: 1, padding: '40px', overflowY: 'auto', position: 'relative' }}>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/demographics" element={<Demographics />} />
          <Route path="/pathogen" element={<Pathogen />} />
          <Route path="/simulation" element={<Simulation />} />
        </Routes>
      </main>
    </Router>
  );
}

export default App;
