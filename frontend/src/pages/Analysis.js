
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Search, 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  FileText, 
  Activity, 
  Eye,
  Zap,
  Target,
  Database,
  Clock,
  TrendingUp,
  Download,
  Share,
  RefreshCw,
  Settings,
  Filter,
  BarChart3
} from 'lucide-react';

// Import components
import APKUpload from '../components/APKUpload';
import AnalysisResults from '../components/AnalysisResults';
import RiskGauge from '../components/RiskGauge';
import EvidenceReport from '../components/EvidenceReport';

const Analysis = () => {
  const [currentScanResults, setCurrentScanResults] = useState(null);
  const [scanHistory, setScanHistory] = useState([]);
  const [activeTab, setActiveTab] = useState('upload');
  const [isScanning, setIsScanning] = useState(false);
  const [analysisMode, setAnalysisMode] = useState('standard');
  const [realtimeStats, setRealtimeStats] = useState({
    activeScan: false,
    queueLength: 0,
    avgScanTime: 8.4,
    successRate: 99.7
  });

  // Initialize with sample scan history
  useEffect(() => {
    const sampleHistory = [
      {
        id: 1,
        fileName: 'banking_app_v2.1.apk',
        scanTime: new Date(Date.now() - 3600000), // 1 hour ago
        riskScore: 12,
        status: 'clean',
        threatLevel: 'low',
        fileSize: 15.2 * 1024 * 1024
      },
      {
        id: 2,
        fileName: 'social_messenger.apk',
        scanTime: new Date(Date.now() - 7200000), // 2 hours ago
        riskScore: 68,
        status: 'suspicious',
        threatLevel: 'medium',
        fileSize: 28.7 * 1024 * 1024
      },
      {
        id: 3,
        fileName: 'game_casino.apk',
        scanTime: new Date(Date.now() - 10800000), // 3 hours ago
        riskScore: 89,
        status: 'malicious',
        threatLevel: 'critical',
        fileSize: 45.1 * 1024 * 1024
      }
    ];
    setScanHistory(sampleHistory);
  }, []);

  // Update real-time stats
  useEffect(() => {
    const interval = setInterval(() => {
      setRealtimeStats(prev => ({
        ...prev,
        queueLength: Math.max(0, prev.queueLength + (Math.random() > 0.7 ? 1 : -1)),
        avgScanTime: 8.4 + (Math.random() - 0.5) * 2,
        successRate: 99.7 + (Math.random() - 0.5) * 0.4
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // Handle scan completion
  const handleScanComplete = (results) => {
    setCurrentScanResults(results);
    setActiveTab('results');
    
    // Add to scan history
    const newHistoryItem = {
      id: Date.now(),
      fileName: results.fileName,
      scanTime: results.scanTime,
      riskScore: results.riskScore,
      status: results.status,
      threatLevel: results.threatLevel,
      fileSize: results.fileSize
    };
    
    setScanHistory(prev => [newHistoryItem, ...prev]);
    setIsScanning(false);
  };

  // Handle new scan start
  const handleNewScan = () => {
    setCurrentScanResults(null);
    setActiveTab('upload');
    setIsScanning(true);
    setRealtimeStats(prev => ({ ...prev, activeScan: true }));
  };

  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'clean': return 'text-green-400 bg-green-900/20';
      case 'suspicious': return 'text-yellow-400 bg-yellow-900/20';
      case 'malicious': return 'text-red-400 bg-red-900/20';
      default: return 'text-gray-400 bg-gray-900/20';
    }
  };

  // Get threat level color
  const getThreatColor = (level) => {
    switch (level) {
      case 'low': return 'text-green-400';
      case 'medium': return 'text-yellow-400';
      case 'high': return 'text-orange-400';
      case 'critical': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const tabs = [
    { 
      id: 'upload', 
      name: 'Upload & Scan', 
      icon: Search,
      description: 'Upload APK files for analysis'
    },
    { 
      id: 'results', 
      name: 'Results', 
      icon: BarChart3,
      description: 'View detailed analysis results',
      disabled: !currentScanResults
    },
    { 
      id: 'evidence', 
      name: 'Evidence', 
      icon: FileText,
      description: 'Security evidence and findings',
      disabled: !currentScanResults
    },
    { 
      id: 'history', 
      name: 'History', 
      icon: Clock,
      description: 'Previous scan results'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6 pt-24">
      <div className="max-w-7xl mx-auto">
        
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2 flex items-center space-x-3">
                <div className="p-2 bg-blue-900/30 rounded-lg">
                  <Search className="w-8 h-8 text-blue-400" />
                </div>
                <span>APK Security Analysis</span>
              </h1>
              <p className="text-gray-400">
                Advanced threat detection and security assessment for Android applications
              </p>
            </div>
            
            {/* Analysis Mode Selector */}
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Settings className="w-4 h-4 text-gray-400" />
                <select
                  value={analysisMode}
                  onChange={(e) => setAnalysisMode(e.target.value)}
                  className="bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500"
                >
                  <option value="standard">Standard Analysis</option>
                  <option value="deep">Deep Quantum Scan</option>
                  <option value="behavioral">Behavioral Analysis</option>
                  <option value="network">Network Forensics</option>
                </select>
              </div>
              
              {currentScanResults && (
                <motion.button
                  onClick={handleNewScan}
                  className="flex items-center space-x-2 bg-green-600 hover:bg-green-500 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>New Scan</span>
                </motion.button>
              )}
            </div>
          </div>
        </motion.div>

        {/* Real-time Status Bar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-gray-800 border border-gray-700 rounded-xl p-4 mb-8"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${realtimeStats.activeScan ? 'bg-green-400 animate-pulse' : 'bg-gray-500'}`}></div>
                <span className="text-sm text-gray-300">
                  {realtimeStats.activeScan ? 'Analysis Active' : 'System Ready'}
                </span>
              </div>
              <div className="text-sm text-gray-400">
                Queue: <span className="text-white font-medium">{realtimeStats.queueLength}</span>
              </div>
              <div className="text-sm text-gray-400">
                Avg Time: <span className="text-white font-medium">{realtimeStats.avgScanTime.toFixed(1)}s</span>
              </div>
              <div className="text-sm text-gray-400">
                Success Rate: <span className="text-green-400 font-medium">{realtimeStats.successRate.toFixed(1)}%</span>
              </div>
            </div>
            
            <div className="flex items-center space-x-4 text-sm">
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-blue-400" />
                <span className="text-gray-400">Quantum Engine: </span>
                <span className="text-blue-400 font-medium">Online</span>
              </div>
              <div className="flex items-center space-x-2">
                <Shield className="w-4 h-4 text-green-400" />
                <span className="text-gray-400">Threat DB: </span>
                <span className="text-green-400 font-medium">Updated</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Navigation Tabs */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="mb-8"
        >
          <div className="flex space-x-1 bg-gray-800 p-1 rounded-xl border border-gray-700">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              const isDisabled = tab.disabled;
              
              return (
                <button
                  key={tab.id}
                  onClick={() => !isDisabled && setActiveTab(tab.id)}
                  disabled={isDisabled}
                  className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-all relative group ${
                    isActive 
                      ? 'bg-gradient-to-r from-green-600 to-blue-600 text-white shadow-lg' 
                      : isDisabled
                      ? 'text-gray-500 cursor-not-allowed'
                      : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.name}</span>
                  
                  {/* Tooltip */}
                  {!isActive && (
                    <div className="absolute top-full mt-2 left-1/2 transform -translate-x-1/2 px-3 py-2 bg-gray-900 border border-gray-700 text-xs text-white rounded-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-10">
                      {tab.description}
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        </motion.div>

        {/* Tab Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            
            {/* Upload & Scan Tab */}
            {activeTab === 'upload' && (
              <div className="space-y-8">
                <APKUpload onScanComplete={handleScanComplete} />
                
                {/* Analysis Tips */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5 }}
                  className="bg-gray-800/50 border border-gray-700 rounded-xl p-6"
                >
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
                    <Eye className="w-5 h-5 text-blue-400" />
                    <span>Analysis Features</span>
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {[
                      {
                        icon: Zap,
                        title: 'Quantum Detection',
                        description: 'Advanced quantum algorithms detect sophisticated threats',
                        color: 'text-blue-400'
                      },
                      {
                        icon: Target,
                        title: 'Zero-Day Protection',
                        description: 'Identify previously unknown malware variants',
                        color: 'text-green-400'
                      },
                      {
                        icon: Database,
                        title: 'Threat Intelligence',
                        description: 'Real-time updates from global security network',
                        color: 'text-purple-400'
                      }
                    ].map((feature, index) => (
                      <motion.div
                        key={feature.title}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.6 + index * 0.1 }}
                        className="text-center p-4 bg-gray-700/30 rounded-lg"
                      >
                        <feature.icon className={`w-8 h-8 ${feature.color} mx-auto mb-3`} />
                        <h4 className="font-medium text-white mb-2">{feature.title}</h4>
                        <p className="text-sm text-gray-400">{feature.description}</p>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              </div>
            )}

            {/* Results Tab */}
            {activeTab === 'results' && currentScanResults && (
              <div className="space-y-8">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                  {/* Risk Gauge */}
                  <div className="lg:col-span-1">
                    <RiskGauge 
                      riskScore={currentScanResults.riskScore}
                      title="Risk Assessment"
                      showDetails={true}
                      size={280}
                    />
                  </div>
                  
                  {/* Quick Stats */}
                  <div className="lg:col-span-2 space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
                        <div className="flex items-center space-x-3 mb-2">
                          <CheckCircle className="w-5 h-5 text-green-400" />
                          <span className="text-sm text-gray-400">Certificate Status</span>
                        </div>
                        <div className={`text-lg font-semibold ${
                          currentScanResults.metrics?.certificateValid ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {currentScanResults.metrics?.certificateValid ? 'Valid' : 'Invalid'}
                        </div>
                      </div>
                      
                      <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
                        <div className="flex items-center space-x-3 mb-2">
                          <AlertTriangle className="w-5 h-5 text-yellow-400" />
                          <span className="text-sm text-gray-400">Findings</span>
                        </div>
                        <div className="text-lg font-semibold text-white">
                          {currentScanResults.findings?.length || 0} Issues
                        </div>
                      </div>
                      
                      <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
                        <div className="flex items-center space-x-3 mb-2">
                          <Activity className="w-5 h-5 text-blue-400" />
                          <span className="text-sm text-gray-400">Code Integrity</span>
                        </div>
                        <div className="text-lg font-semibold text-blue-400">
                          {currentScanResults.metrics?.codeIntegrity || 0}%
                        </div>
                      </div>
                      
                      <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
                        <div className="flex items-center space-x-3 mb-2">
                          <Zap className="w-5 h-5 text-purple-400" />
                          <span className="text-sm text-gray-400">Quantum Safe</span>
                        </div>
                        <div className={`text-lg font-semibold ${
                          currentScanResults.quantum?.quantumResistant ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {currentScanResults.quantum?.quantumResistant ? 'Yes' : 'No'}
                        </div>
                      </div>
                    </div>
                    
                    {/* Action Buttons */}
                    <div className="flex items-center space-x-4">
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-500 text-white px-6 py-3 rounded-lg font-medium transition-colors"
                      >
                        <Download className="w-4 h-4" />
                        <span>Export Report</span>
                      </motion.button>
                      
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="flex items-center space-x-2 bg-gray-600 hover:bg-gray-500 text-white px-6 py-3 rounded-lg font-medium transition-colors"
                      >
                        <Share className="w-4 h-4" />
                        <span>Share Results</span>
                      </motion.button>
                      
                      <motion.button
                        onClick={() => setActiveTab('evidence')}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="flex items-center space-x-2 bg-green-600 hover:bg-green-500 text-white px-6 py-3 rounded-lg font-medium transition-colors"
                      >
                        <FileText className="w-4 h-4" />
                        <span>View Evidence</span>
                      </motion.button>
                    </div>
                  </div>
                </div>
                
                {/* Detailed Analysis Results */}
                <AnalysisResults results={currentScanResults} />
              </div>
            )}

            {/* Evidence Tab */}
            {activeTab === 'evidence' && currentScanResults && (
              <div>
                <EvidenceReport 
                  scanResults={currentScanResults}
                  showHeader={true}
                  maxHeight="800px"
                />
              </div>
            )}

            {/* History Tab */}
            {activeTab === 'history' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-xl font-semibold text-white">Scan History</h3>
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2">
                      <Filter className="w-4 h-4 text-gray-400" />
                      <select className="bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500">
                        <option>All Results</option>
                        <option>Clean Only</option>
                        <option>Threats Only</option>
                        <option>High Risk</option>
                      </select>
                    </div>
                    <span className="text-sm text-gray-400">
                      {scanHistory.length} total scans
                    </span>
                  </div>
                </div>
                
                <div className="space-y-4">
                  {scanHistory.map((scan, index) => (
                    <motion.div
                      key={scan.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="bg-gray-800 border border-gray-700 rounded-xl p-6 hover:bg-gray-700/50 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                          <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                            scan.status === 'clean' ? 'bg-green-900/30' :
                            scan.status === 'suspicious' ? 'bg-yellow-900/30' :
                            'bg-red-900/30'
                          }`}>
                            {scan.status === 'clean' ? (
                              <CheckCircle className="w-6 h-6 text-green-400" />
                            ) : scan.status === 'suspicious' ? (
                              <AlertTriangle className="w-6 h-6 text-yellow-400" />
                            ) : (
                              <AlertTriangle className="w-6 h-6 text-red-400" />
                            )}
                          </div>
                          <div>
                            <h4 className="text-lg font-medium text-white">{scan.fileName}</h4>
                            <div className="flex items-center space-x-4 text-sm text-gray-400">
                              <span>{scan.scanTime.toLocaleString()}</span>
                              <span>{(scan.fileSize / (1024 * 1024)).toFixed(1)} MB</span>
                              <span className={getThreatColor(scan.threatLevel)}>
                                {scan.threatLevel.toUpperCase()} RISK
                              </span>
                            </div>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-6">
                          <div className="text-right">
                            <div className="text-2xl font-bold text-white">{scan.riskScore}%</div>
                            <div className={`text-sm font-medium ${getStatusColor(scan.status)}`}>
                              {scan.status.toUpperCase()}
                            </div>
                          </div>
                          
                          <motion.button
                            onClick={() => {
                              // In a real app, you would load the full results
                              setCurrentScanResults({
                                ...scan,
                                metrics: { certificateValid: scan.riskScore < 50 },
                                quantum: { quantumResistant: scan.riskScore < 30 },
                                findings: scan.riskScore > 60 ? ['Sample finding'] : []
                              });
                              setActiveTab('results');
                            }}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                          >
                            <Eye className="w-4 h-4" />
                            <span>View</span>
                          </motion.button>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
                
                {scanHistory.length === 0 && (
                  <div className="text-center py-12">
                    <Clock className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                    <p className="text-gray-400">No scan history available</p>
                    <p className="text-sm text-gray-500">Upload and scan APK files to build your history</p>
                  </div>
                )}
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
};

export default Analysis;
