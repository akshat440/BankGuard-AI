import React, { useState, useEffect } from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import './App.css';

// Components
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';

// Pages
import Home from './pages/Home';
import Analysis from './pages/Analysis';
import Reports from './pages/Reports';

// Utils
import { initializeQuantumSecurity } from './utils/helpers';

// Animation variants for page transitions
const pageVariants = {
  initial: {
    opacity: 0,
    x: -20,
    scale: 0.98
  },
  in: {
    opacity: 1,
    x: 0,
    scale: 1
  },
  out: {
    opacity: 0,
    x: 20,
    scale: 0.98
  }
};

const pageTransition = {
  type: 'tween',
  ease: 'anticipate',
  duration: 0.3
};

// Main App Component
function App() {
  const [isSecurityInitialized, setIsSecurityInitialized] = useState(false);
  const [systemStatus, setSystemStatus] = useState('initializing');
  const [threatLevel, setThreatLevel] = useState('low');
  const [globalStats, setGlobalStats] = useState({
    appsProtected: 0,
    threatsToday: 0,
    breaches: 0,
    quantumShield: false
  });

  const location = useLocation();

  // Initialize quantum security system
  useEffect(() => {
    const initSecurity = async () => {
      try {
        setSystemStatus('initializing');
        
        // Simulate quantum security initialization
        await initializeQuantumSecurity();
        
        setSystemStatus('active');
        setIsSecurityInitialized(true);
        
        // Initialize global stats with animation
        animateGlobalStats();
        
        // Simulate quantum shield activation
        setTimeout(() => {
          setGlobalStats(prev => ({ ...prev, quantumShield: true }));
        }, 2500);
        
      } catch (error) {
        console.error('❌ Quantum security initialization failed:', error);
        setSystemStatus('error');
      }
    };

    initSecurity();
    
    // Set up threat level monitoring
    const threatMonitor = setInterval(() => {
      updateThreatLevel();
    }, 30000); // Update every 30 seconds

    // Set up real-time stats updates
    const statsUpdater = setInterval(() => {
      updateRealTimeStats();
    }, 5000); // Update every 5 seconds

    return () => {
      clearInterval(threatMonitor);
      clearInterval(statsUpdater);
    };
  }, []);

  // Animate global stats on initialization
  const animateGlobalStats = () => {
    // Apps Protected counter
    let appsCount = 0;
    const appsInterval = setInterval(() => {
      appsCount += Math.floor(Math.random() * 50) + 10;
      if (appsCount >= 12847) {
        appsCount = 12847;
        clearInterval(appsInterval);
      }
      setGlobalStats(prev => ({ ...prev, appsProtected: appsCount }));
    }, 50);

    // Threats Today counter
    setTimeout(() => {
      let threatsCount = 0;
      const threatsInterval = setInterval(() => {
        threatsCount += Math.floor(Math.random() * 5) + 1;
        if (threatsCount >= 247) {
          threatsCount = 247;
          clearInterval(threatsInterval);
        }
        setGlobalStats(prev => ({ ...prev, threatsToday: threatsCount }));
      }, 80);
    }, 500);
  };

  // Update threat level based on simulation
  const updateThreatLevel = () => {
    const levels = ['low', 'medium', 'high', 'critical'];
    const weights = [0.6, 0.25, 0.1, 0.05]; // Weighted random
    
    let random = Math.random();
    let selectedLevel = 'low';
    
    for (let i = 0; i < levels.length; i++) {
      if (random < weights[i]) {
        selectedLevel = levels[i];
        break;
      }
      random -= weights[i];
    }
    
    setThreatLevel(selectedLevel);
    
    // Log threat level changes
    if (selectedLevel !== 'low') {
      console.log(`🚨 Threat level elevated to: ${selectedLevel.toUpperCase()}`);
    }
  };

  // Update real-time statistics
  const updateRealTimeStats = () => {
    setGlobalStats(prev => ({
      ...prev,
      appsProtected: prev.appsProtected + Math.floor(Math.random() * 3),
      threatsToday: prev.threatsToday + Math.floor(Math.random() * 2)
    }));
  };

  // Get threat level color
  const getThreatLevelColor = (level) => {
    switch (level) {
      case 'low': return 'text-green-400';
      case 'medium': return 'text-yellow-400';
      case 'high': return 'text-orange-400';
      case 'critical': return 'text-red-400';
      default: return 'text-green-400';
    }
  };

  // Loading screen component
  if (!isSecurityInitialized && systemStatus === 'initializing') {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <motion.div 
          className="text-center"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <motion.div
            className="w-20 h-20 border-4 border-green-400 border-t-transparent rounded-full mx-auto mb-6"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          />
          <h2 className="text-2xl font-bold text-green-400 mb-2">
            Quantum Security Initializing
          </h2>
          <motion.p 
            className="text-gray-400"
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            Deploying quantum threat detection protocols...
          </motion.p>
          <div className="mt-6 text-sm text-gray-500 space-y-1">
            <div>⚡ Quantum encryption: {systemStatus === 'active' ? '✅' : '⏳'}</div>
            <div>🛡️ Shield protocols: {globalStats.quantumShield ? '✅' : '⏳'}</div>
            <div>🔍 Threat detection: {isSecurityInitialized ? '✅' : '⏳'}</div>
          </div>
        </motion.div>
      </div>
    );
  }

  // Error state
  if (systemStatus === 'error') {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <motion.div 
          className="text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <div className="text-red-400 text-6xl mb-4">🚨</div>
          <h1 className="text-2xl font-bold text-red-400 mb-4">
            Quantum Security System Error
          </h1>
          <p className="text-gray-300 mb-6">
            Unable to initialize quantum security protocols
          </p>
          <button
            onClick={() => window.location.reload()}
            className="bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg transition-colors"
          >
            Restart System
          </button>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Global Quantum Grid Background */}
      <div className="fixed inset-0 opacity-5 pointer-events-none">
        <div 
          className="w-full h-full" 
          style={{
            backgroundImage: `
              linear-gradient(rgba(0, 255, 136, 0.1) 1px, transparent 1px),
              linear-gradient(90deg, rgba(0, 255, 136, 0.1) 1px, transparent 1px)
            `,
            backgroundSize: '50px 50px'
          }}
        />
      </div>

      {/* System Status Bar */}
      <motion.div 
        className="fixed top-0 left-0 right-0 bg-gray-800/80 backdrop-blur-sm border-b border-gray-700 z-50"
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <div className="max-w-7xl mx-auto px-4 py-2 flex items-center justify-between text-sm">
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-green-400 font-medium">QUANTUM ACTIVE</span>
            </div>
            <div className="text-gray-400">
              Threat Level: <span className={`font-medium ${getThreatLevelColor(threatLevel)}`}>
                {threatLevel.toUpperCase()}
              </span>
            </div>
          </div>
          
          <div className="flex items-center space-x-6 text-gray-400">
            <div>Protected: <span className="text-green-400 font-medium">
              {globalStats.appsProtected.toLocaleString()}
            </span></div>
            <div>Threats Today: <span className="text-yellow-400 font-medium">
              {globalStats.threatsToday}
            </span></div>
            <div>Breaches: <span className="text-green-400 font-medium">
              {globalStats.breaches}
            </span></div>
            {globalStats.quantumShield && (
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                <span className="text-blue-400">QUANTUM SHIELD</span>
              </div>
            )}
          </div>
        </div>
      </motion.div>

      {/* Navigation */}
      <Navbar />

      {/* Main Content */}
      <main className="pt-20">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial="initial"
            animate="in"
            exit="out"
            variants={pageVariants}
            transition={pageTransition}
            className="min-h-screen"
          >
            <Routes>
              <Route path="/" element={<Home globalStats={globalStats} threatLevel={threatLevel} />} />
              <Route path="/dashboard" element={<Dashboard globalStats={globalStats} threatLevel={threatLevel} />} />
              <Route path="/analysis" element={<Analysis />} />
              <Route path="/reports" element={<Reports />} />
              <Route path="*" element={
                <div className="min-h-screen flex items-center justify-center">
                  <div className="text-center">
                    <h1 className="text-4xl font-bold text-red-400 mb-4">404</h1>
                    <p className="text-gray-400">Quantum pathway not found</p>
                  </div>
                </div>
              } />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Quantum Particles Background Effect */}
      <motion.div
        className="fixed inset-0 pointer-events-none z-0"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 2, delay: 1 }}
      >
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-green-400 rounded-full opacity-20"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`
            }}
            animate={{
              x: [0, Math.random() * 100 - 50],
              y: [0, Math.random() * 100 - 50],
              opacity: [0.2, 0.8, 0.2]
            }}
            transition={{
              duration: Math.random() * 10 + 10,
              repeat: Infinity,
              repeatType: 'reverse'
            }}
          />
        ))}
      </motion.div>
    </div>
  );
}

export default App;
