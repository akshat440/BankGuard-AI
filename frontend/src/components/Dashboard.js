import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Shield, 
  AlertTriangle, 
  Activity, 
  Users, 
  TrendingUp, 
  Eye, 
  Zap, 
  Lock,
  Globe,
  Cpu,
  Database,
  Wifi,
  Server,
  Clock,
  Target
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

const Dashboard = ({ globalStats, threatLevel }) => {
  const [threatIntelligence, setThreatIntelligence] = useState([]);
  const [systemMetrics, setSystemMetrics] = useState({});
  const [networkActivity, setNetworkActivity] = useState([]);
  const [recentScans, setRecentScans] = useState([]);
  const [currentThreat, setCurrentThreat] = useState(0);
  const [isQuantumActive, setIsQuantumActive] = useState(true);

  // Initialize dashboard data
  useEffect(() => {
    initializeThreatIntelligence();
    initializeSystemMetrics();
    initializeNetworkActivity();
    initializeRecentScans();
    
    // Set up real-time updates
    const threatRotator = setInterval(() => {
      setCurrentThreat(prev => (prev + 1) % threatIntelligence.length);
    }, 30000); // Rotate every 30 seconds

    const metricsUpdater = setInterval(() => {
      updateSystemMetrics();
    }, 5000); // Update every 5 seconds

    const networkUpdater = setInterval(() => {
      updateNetworkActivity();
    }, 3000); // Update every 3 seconds

    return () => {
      clearInterval(threatRotator);
      clearInterval(metricsUpdater);
      clearInterval(networkUpdater);
    };
  }, [threatIntelligence.length]);

  // Initialize threat intelligence data
  const initializeThreatIntelligence = () => {
    const threats = [
      {
        id: 1,
        name: 'QUANTUM_MALWARE_X7',
        severity: 'critical',
        type: 'Advanced Persistent Threat',
        description: 'Sophisticated banking trojan with quantum encryption bypass',
        impact: 'High',
        firstSeen: '2024-08-29',
        affectedApps: 847,
        origin: 'Eastern Europe'
      },
      {
        id: 2,
        name: 'NEURAL_SPYWARE_V3',
        severity: 'high',
        type: 'Spyware Campaign',
        description: 'AI-powered data exfiltration targeting financial apps',
        impact: 'Medium',
        firstSeen: '2024-08-30',
        affectedApps: 412,
        origin: 'Unknown'
      },
      {
        id: 3,
        name: 'CRYPTO_MINER_SWARM',
        severity: 'medium',
        type: 'Cryptojacking',
        description: 'Distributed cryptocurrency mining through infected APKs',
        impact: 'Low',
        firstSeen: '2024-08-31',
        affectedApps: 1247,
        origin: 'Asia-Pacific'
      },
      {
        id: 4,
        name: 'PHANTOM_ROOTKIT_2.0',
        severity: 'critical',
        type: 'Rootkit',
        description: 'Stealth rootkit with anti-detection quantum algorithms',
        impact: 'Critical',
        firstSeen: '2024-08-28',
        affectedApps: 89,
        origin: 'Classified'
      },
      {
        id: 5,
        name: 'SOCIAL_HARVESTER_PRO',
        severity: 'high',
        type: 'Data Harvester',
        description: 'Social media credential theft with ML-powered attacks',
        impact: 'High',
        firstSeen: '2024-08-30',
        affectedApps: 634,
        origin: 'North America'
      }
    ];
    setThreatIntelligence(threats);
  };

  // Initialize system metrics
  const initializeSystemMetrics = () => {
    setSystemMetrics({
      cpuUsage: 23 + Math.random() * 10,
      memoryUsage: 67 + Math.random() * 15,
      diskUsage: 45 + Math.random() * 20,
      networkLatency: 12 + Math.random() * 8,
      quantumEfficiency: 94 + Math.random() * 5,
      threatsBlocked: 1847 + Math.floor(Math.random() * 100),
      scanSpeed: 2.4 + Math.random() * 0.8,
      uptime: '15d 7h 42m'
    });
  };

  // Initialize network activity data
  const initializeNetworkActivity = () => {
    const data = [];
    const now = new Date();
    for (let i = 23; i >= 0; i--) {
      const time = new Date(now.getTime() - i * 60 * 60 * 1000);
      data.push({
        time: time.getHours() + ':00',
        threats: Math.floor(Math.random() * 50) + 10,
        scans: Math.floor(Math.random() * 200) + 50,
        blocked: Math.floor(Math.random() * 30) + 5
      });
    }
    setNetworkActivity(data);
  };

  // Initialize recent scans
  const initializeRecentScans = () => {
    const apps = [
      { name: 'banking_app.apk', result: 'clean', time: '2 min ago', risk: 0 },
      { name: 'social_media.apk', result: 'suspicious', time: '5 min ago', risk: 65 },
      { name: 'game_puzzle.apk', result: 'malicious', time: '8 min ago', risk: 94 },
      { name: 'messaging_pro.apk', result: 'clean', time: '12 min ago', risk: 12 },
      { name: 'photo_editor.apk', result: 'clean', time: '15 min ago', risk: 3 },
      { name: 'crypto_wallet.apk', result: 'suspicious', time: '18 min ago', risk: 78 }
    ];
    setRecentScans(apps);
  };

  // Update system metrics
  const updateSystemMetrics = () => {
    setSystemMetrics(prev => ({
      ...prev,
      cpuUsage: Math.max(15, Math.min(85, prev.cpuUsage + (Math.random() - 0.5) * 10)),
      memoryUsage: Math.max(40, Math.min(90, prev.memoryUsage + (Math.random() - 0.5) * 8)),
      networkLatency: Math.max(5, Math.min(50, prev.networkLatency + (Math.random() - 0.5) * 5)),
      threatsBlocked: prev.threatsBlocked + Math.floor(Math.random() * 3)
    }));
  };

  // Update network activity
  const updateNetworkActivity = () => {
    setNetworkActivity(prev => {
      const newData = [...prev.slice(1)];
      const now = new Date();
      newData.push({
        time: now.getHours() + ':' + String(now.getMinutes()).padStart(2, '0'),
        threats: Math.floor(Math.random() * 50) + 10,
        scans: Math.floor(Math.random() * 200) + 50,
        blocked: Math.floor(Math.random() * 30) + 5
      });
      return newData;
    });
  };

  // Get severity color
  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical': return 'text-red-400 bg-red-900/20 border-red-500/30';
      case 'high': return 'text-orange-400 bg-orange-900/20 border-orange-500/30';
      case 'medium': return 'text-yellow-400 bg-yellow-900/20 border-yellow-500/30';
      case 'low': return 'text-green-400 bg-green-900/20 border-green-500/30';
      default: return 'text-gray-400 bg-gray-900/20 border-gray-500/30';
    }
  };

  // Get scan result color
  const getScanResultColor = (result) => {
    switch (result) {
      case 'clean': return 'text-green-400 bg-green-900/20';
      case 'suspicious': return 'text-yellow-400 bg-yellow-900/20';
      case 'malicious': return 'text-red-400 bg-red-900/20';
      default: return 'text-gray-400 bg-gray-900/20';
    }
  };

  // Pie chart data for threat distribution
  const threatDistribution = [
    { name: 'Malware', value: 35, color: '#ef4444' },
    { name: 'Spyware', value: 25, color: '#f97316' },
    { name: 'Adware', value: 20, color: '#eab308' },
    { name: 'Trojan', value: 15, color: '#8b5cf6' },
    { name: 'Other', value: 5, color: '#6b7280' }
  ];

  return (
    <div className="min-h-screen bg-gray-900 p-6 pt-24">
      <div className="max-w-7xl mx-auto">
        
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-white mb-2">
            Quantum Security Command Center
          </h1>
          <p className="text-gray-400">
            Real-time threat monitoring and system analytics
          </p>
        </motion.div>

        {/* Main Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          
          {/* Apps Protected */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
            className="bg-gray-800 border border-gray-700 rounded-xl p-6 hover:bg-gray-700/50 transition-all duration-300"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-green-900/30 rounded-lg">
                <Shield className="w-6 h-6 text-green-400" />
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-white">
                  {globalStats.appsProtected?.toLocaleString()}
                </div>
                <div className="text-sm text-gray-400">Apps Protected</div>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <TrendingUp className="w-4 h-4 text-green-400" />
              <span className="text-sm text-green-400">+12% from last hour</span>
            </div>
          </motion.div>

          {/* Threats Today */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="bg-gray-800 border border-gray-700 rounded-xl p-6 hover:bg-gray-700/50 transition-all duration-300"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-yellow-900/30 rounded-lg">
                <AlertTriangle className="w-6 h-6 text-yellow-400" />
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-white">
                  {globalStats.threatsToday}
                </div>
                <div className="text-sm text-gray-400">Threats Detected</div>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Activity className="w-4 h-4 text-yellow-400" />
              <span className="text-sm text-yellow-400">Active monitoring</span>
            </div>
          </motion.div>

          {/* Security Breaches */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
            className="bg-gray-800 border border-gray-700 rounded-xl p-6 hover:bg-gray-700/50 transition-all duration-300"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-green-900/30 rounded-lg">
                <Lock className="w-6 h-6 text-green-400" />
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-green-400">
                  {globalStats.breaches}
                </div>
                <div className="text-sm text-gray-400">Security Breaches</div>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Shield className="w-4 h-4 text-green-400" />
              <span className="text-sm text-green-400">100% Protected</span>
            </div>
          </motion.div>

          {/* Quantum Efficiency */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.4 }}
            className="bg-gray-800 border border-gray-700 rounded-xl p-6 hover:bg-gray-700/50 transition-all duration-300"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-blue-900/30 rounded-lg">
                <Zap className="w-6 h-6 text-blue-400" />
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-blue-400">
                  {systemMetrics.quantumEfficiency?.toFixed(1)}%
                </div>
                <div className="text-sm text-gray-400">Quantum Efficiency</div>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${isQuantumActive ? 'bg-blue-400 animate-pulse' : 'bg-gray-500'}`}></div>
              <span className="text-sm text-blue-400">Quantum Active</span>
            </div>
          </motion.div>
        </div>

        {/* Charts and Intelligence Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          
          {/* Network Activity Chart */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 }}
            className="lg:col-span-2 bg-gray-800 border border-gray-700 rounded-xl p-6"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-white">Network Activity</h3>
              <div className="flex items-center space-x-4 text-sm">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-green-400 rounded-full"></div>
                  <span className="text-gray-400">Scans</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-red-400 rounded-full"></div>
                  <span className="text-gray-400">Threats</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-blue-400 rounded-full"></div>
                  <span className="text-gray-400">Blocked</span>
                </div>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={networkActivity}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#6b7280" />
                <YAxis stroke="#6b7280" />
                <Line 
                  type="monotone" 
                  dataKey="scans" 
                  stroke="#10b981" 
                  strokeWidth={2}
                  dot={{ fill: '#10b981', strokeWidth: 2, r: 4 }}
                  activeDot={{ r: 6 }}
                />
                <Line 
                  type="monotone" 
                  dataKey="threats" 
                  stroke="#ef4444" 
                  strokeWidth={2}
                  dot={{ fill: '#ef4444', strokeWidth: 2, r: 4 }}
                />
                <Line 
                  type="monotone" 
                  dataKey="blocked" 
                  stroke="#3b82f6" 
                  strokeWidth={2}
                  dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Threat Intelligence Center */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.6 }}
            className="bg-gray-800 border border-gray-700 rounded-xl p-6"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-white">Threat Intelligence</h3>
              <Target className="w-5 h-5 text-red-400" />
            </div>
            
            <AnimatePresence mode="wait">
              {threatIntelligence[currentThreat] && (
                <motion.div
                  key={currentThreat}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.5 }}
                  className={`p-4 rounded-lg border ${getSeverityColor(threatIntelligence[currentThreat].severity)} mb-4`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-semibold text-white text-sm">
                      {threatIntelligence[currentThreat].name}
                    </h4>
                    <div className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(threatIntelligence[currentThreat].severity)}`}>
                      {threatIntelligence[currentThreat].severity.toUpperCase()}
                    </div>
                  </div>
                  <p className="text-gray-300 text-sm mb-3">
                    {threatIntelligence[currentThreat].description}
                  </p>
                  <div className="space-y-2 text-xs text-gray-400">
                    <div className="flex justify-between">
                      <span>Type:</span>
                      <span className="text-white">{threatIntelligence[currentThreat].type}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Affected Apps:</span>
                      <span className="text-red-400">{threatIntelligence[currentThreat].affectedApps}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Origin:</span>
                      <span className="text-yellow-400">{threatIntelligence[currentThreat].origin}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>First Seen:</span>
                      <span className="text-gray-300">{threatIntelligence[currentThreat].firstSeen}</span>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="flex justify-center space-x-2 mb-4">
              {threatIntelligence.map((_, index) => (
                <button
                  key={index}
                  onClick={() => setCurrentThreat(index)}
                  className={`w-2 h-2 rounded-full transition-all ${
                    index === currentThreat ? 'bg-red-400' : 'bg-gray-600'
                  }`}
                />
              ))}
            </div>

            <div className="text-center">
              <div className="text-xs text-gray-500">Auto-rotating every 30s</div>
            </div>
          </motion.div>
        </div>

        {/* Bottom Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          
          {/* Recent Scans */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="bg-gray-800 border border-gray-700 rounded-xl p-6"
          >
            <h3 className="text-lg font-semibold text-white mb-6">Recent Scans</h3>
            <div className="space-y-4">
              {recentScans.map((scan, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.8 + index * 0.1 }}
                  className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg hover:bg-gray-700 transition-colors"
                >
                  <div className="flex items-center space-x-3">
                    <div className={`w-3 h-3 rounded-full ${getScanResultColor(scan.result)}`}></div>
                    <div>
                      <div className="text-sm font-medium text-white">{scan.name}</div>
                      <div className="text-xs text-gray-400">{scan.time}</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`text-sm font-medium ${getScanResultColor(scan.result)}`}>
                      {scan.result.toUpperCase()}
                    </div>
                    <div className="text-xs text-gray-400">Risk: {scan.risk}%</div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* System Metrics */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            className="bg-gray-800 border border-gray-700 rounded-xl p-6"
          >
            <h3 className="text-lg font-semibold text-white mb-6">System Metrics</h3>
            <div className="space-y-4">
              
              {/* CPU Usage */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <Cpu className="w-5 h-5 text-blue-400" />
                  <span className="text-sm text-gray-300">CPU Usage</span>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium text-white">{systemMetrics.cpuUsage?.toFixed(1)}%</div>
                  <div className="w-20 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-blue-400 transition-all duration-500"
                      style={{ width: `${systemMetrics.cpuUsage}%` }}
                    ></div>
                  </div>
                </div>
              </div>

              {/* Memory Usage */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <Database className="w-5 h-5 text-green-400" />
                  <span className="text-sm text-gray-300">Memory</span>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium text-white">{systemMetrics.memoryUsage?.toFixed(1)}%</div>
                  <div className="w-20 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-green-400 transition-all duration-500"
                      style={{ width: `${systemMetrics.memoryUsage}%` }}
                    ></div>
                  </div>
                </div>
              </div>

              {/* Network Latency */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <Wifi className="w-5 h-5 text-yellow-400" />
                  <span className="text-sm text-gray-300">Network Latency</span>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium text-white">{systemMetrics.networkLatency?.toFixed(1)}ms</div>
                </div>
              </div>

              {/* Threats Blocked */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <Shield className="w-5 h-5 text-red-400" />
                  <span className="text-sm text-gray-300">Threats Blocked</span>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium text-red-400">{systemMetrics.threatsBlocked}</div>
                </div>
              </div>

              {/* Scan Speed */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <Activity className="w-5 h-5 text-purple-400" />
                  <span className="text-sm text-gray-300">Scan Speed</span>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium text-purple-400">{systemMetrics.scanSpeed?.toFixed(1)}s/app</div>
                </div>
              </div>

              {/* System Uptime */}
              <div className="flex items-center justify-between pt-2 border-t border-gray-700">
                <div className="flex items-center space-x-3">
                  <Clock className="w-5 h-5 text-gray-400" />
                  <span className="text-sm text-gray-300">System Uptime</span>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium text-gray-400">{systemMetrics.uptime}</div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
