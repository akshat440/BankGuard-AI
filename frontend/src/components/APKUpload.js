import React, { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  File, 
  X, 
  Play, 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  Zap,
  Eye,
  Lock,
  Search,
  Activity,
  Database,
  Cpu
} from 'lucide-react';

const APKUpload = ({ onScanComplete }) => {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [scanningFile, setScanningFile] = useState(null);
  const [scanProgress, setScanProgress] = useState(0);
  const [scanPhase, setScanPhase] = useState('');
  const [scanResults, setScanResults] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  
  const fileInputRef = useRef(null);
  const dragCounter = useRef(0);

  // Scanning phases for realistic simulation
  const scanPhases = [
    { name: 'Initializing Quantum Scanner', duration: 800, progress: 5 },
    { name: 'Extracting APK Components', duration: 1200, progress: 15 },
    { name: 'Analyzing Manifest File', duration: 1000, progress: 25 },
    { name: 'Scanning DEX Bytecode', duration: 1500, progress: 40 },
    { name: 'Checking Permissions', duration: 800, progress: 50 },
    { name: 'Analyzing Network Calls', duration: 1200, progress: 65 },
    { name: 'Running Quantum Threat Detection', duration: 2000, progress: 80 },
    { name: 'Cross-referencing Threat Database', duration: 1000, progress: 90 },
    { name: 'Generating Security Report', duration: 800, progress: 95 },
    { name: 'Finalizing Analysis', duration: 500, progress: 100 }
  ];

  // Handle drag events
  const handleDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current++;
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragging(true);
    }
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current--;
    if (dragCounter.current === 0) {
      setIsDragging(false);
    }
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    dragCounter.current = 0;
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(Array.from(e.dataTransfer.files));
      e.dataTransfer.clearData();
    }
  }, []);

  // Handle file selection
  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(Array.from(e.target.files));
    }
  };

  // Process uploaded files
  const handleFiles = (files) => {
    const apkFiles = files.filter(file => 
      file.name.toLowerCase().endsWith('.apk') || 
      file.type === 'application/vnd.android.package-archive'
    );
    
    if (apkFiles.length === 0) {
      alert('Please upload valid APK files only.');
      return;
    }

    const newFiles = apkFiles.map((file, index) => ({
      id: Date.now() + index,
      file,
      name: file.name,
      size: file.size,
      status: 'uploaded',
      uploadTime: new Date()
    }));

    setUploadedFiles(prev => [...prev, ...newFiles]);
  };

  // Remove file from upload list
  const removeFile = (fileId) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
  };

  // Start scanning process
  const startScan = async (fileObj) => {
    if (isScanning) return;
    
    setIsScanning(true);
    setScanningFile(fileObj);
    setScanProgress(0);
    setScanResults(null);
    
    // Update file status
    setUploadedFiles(prev => 
      prev.map(f => f.id === fileObj.id ? { ...f, status: 'scanning' } : f)
    );

    // Simulate scanning phases
    for (let i = 0; i < scanPhases.length; i++) {
      const phase = scanPhases[i];
      setScanPhase(phase.name);
      
      // Animate progress for this phase
      const startProgress = i === 0 ? 0 : scanPhases[i - 1].progress;
      const endProgress = phase.progress;
      const steps = 20;
      const stepDuration = phase.duration / steps;
      
      for (let step = 0; step <= steps; step++) {
        const progress = startProgress + (endProgress - startProgress) * (step / steps);
        setScanProgress(progress);
        await new Promise(resolve => setTimeout(resolve, stepDuration));
      }
    }

    // Generate scan results
    const results = generateScanResults(fileObj);
    setScanResults(results);
    
    // Update file status
    setUploadedFiles(prev => 
      prev.map(f => f.id === fileObj.id ? { 
        ...f, 
        status: 'completed',
        results 
      } : f)
    );

    setIsScanning(false);
    setScanningFile(null);
    setScanPhase('');
    
    // Notify parent component
    if (onScanComplete) {
      onScanComplete(results);
    }
  };

  // Generate realistic scan results
  const generateScanResults = (fileObj) => {
    const riskFactors = [
      'Excessive permissions requested',
      'Obfuscated code detected',
      'Network communication to suspicious domains',
      'Potential data harvesting behavior',
      'Anti-debugging techniques found',
      'Dynamic code loading detected',
      'Root detection bypass attempts',
      'Cryptocurrency mining components',
      'Ad fraud mechanisms',
      'SMS premium service abuse'
    ];

    // Determine threat level based on file name (for demo purposes)
    let threatLevel = 'low';
    let riskScore = Math.random() * 30; // Default low risk
    
    if (fileObj.name.toLowerCase().includes('bank')) {
      threatLevel = Math.random() > 0.7 ? 'high' : 'medium';
      riskScore = Math.random() > 0.7 ? 70 + Math.random() * 25 : 40 + Math.random() * 30;
    } else if (fileObj.name.toLowerCase().includes('game')) {
      threatLevel = Math.random() > 0.8 ? 'medium' : 'low';
      riskScore = Math.random() > 0.8 ? 40 + Math.random() * 30 : Math.random() * 40;
    } else if (fileObj.name.toLowerCase().includes('social')) {
      threatLevel = Math.random() > 0.6 ? 'medium' : 'low';
      riskScore = Math.random() > 0.6 ? 50 + Math.random() * 30 : Math.random() * 50;
    } else if (fileObj.name.toLowerCase().includes('malware') || 
               fileObj.name.toLowerCase().includes('virus') ||
               fileObj.name.toLowerCase().includes('trojan')) {
      threatLevel = 'critical';
      riskScore = 85 + Math.random() * 15;
    }

    // Generate findings based on threat level
    let findings = [];
    const numFindings = threatLevel === 'critical' ? 5 + Math.floor(Math.random() * 3) :
                       threatLevel === 'high' ? 3 + Math.floor(Math.random() * 3) :
                       threatLevel === 'medium' ? 1 + Math.floor(Math.random() * 3) :
                       Math.floor(Math.random() * 2);

    const shuffledFactors = [...riskFactors].sort(() => 0.5 - Math.random());
    findings = shuffledFactors.slice(0, numFindings);

    return {
      fileId: fileObj.id,
      fileName: fileObj.name,
      fileSize: fileObj.size,
      scanTime: new Date(),
      threatLevel,
      riskScore: Math.round(riskScore),
      status: threatLevel === 'critical' ? 'malicious' : 
              threatLevel === 'high' ? 'suspicious' : 
              threatLevel === 'medium' ? 'caution' : 'clean',
      findings,
      metrics: {
        certificateValid: threatLevel === 'critical' ? false : Math.random() > 0.3,
        codeIntegrity: Math.round((100 - riskScore) + Math.random() * 10),
        permissionScore: Math.round(85 - riskScore * 0.5 + Math.random() * 10),
        networkSecurity: Math.round(90 - riskScore * 0.3 + Math.random() * 10),
        dataProtection: Math.round(95 - riskScore * 0.4 + Math.random() * 10)
      },
      quantum: {
        quantumSignature: Math.random().toString(36).substring(2, 15).toUpperCase(),
        entropyScore: Math.round(Math.random() * 100),
        quantumResistant: threatLevel === 'critical' ? false : Math.random() > 0.2
      }
    };
  };

  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'uploaded': return 'text-blue-400 bg-blue-900/20';
      case 'scanning': return 'text-yellow-400 bg-yellow-900/20';
      case 'completed': return 'text-green-400 bg-green-900/20';
      case 'error': return 'text-red-400 bg-red-900/20';
      default: return 'text-gray-400 bg-gray-900/20';
    }
  };

  return (
    <div className="space-y-6">
      
      {/* Upload Zone */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={`relative border-2 border-dashed rounded-xl p-8 transition-all duration-300 ${
          isDragging 
            ? 'border-green-400 bg-green-400/10' 
            : 'border-gray-600 bg-gray-800/50 hover:border-gray-500 hover:bg-gray-800/70'
        }`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".apk,application/vnd.android.package-archive"
          onChange={handleFileSelect}
          className="hidden"
        />
        
        <div className="text-center">
          <motion.div
            animate={{ 
              scale: isDragging ? 1.1 : 1,
              rotate: isDragging ? 5 : 0 
            }}
            transition={{ type: 'spring', stiffness: 300, damping: 20 }}
            className={`mx-auto w-16 h-16 rounded-full flex items-center justify-center mb-4 ${
              isDragging ? 'bg-green-400/20' : 'bg-gray-700/50'
            }`}
          >
            <Upload className={`w-8 h-8 ${isDragging ? 'text-green-400' : 'text-gray-400'}`} />
          </motion.div>
          
          <h3 className="text-lg font-semibold text-white mb-2">
            {isDragging ? 'Drop APK files here' : 'Upload APK Files for Analysis'}
          </h3>
          
          <p className="text-gray-400 mb-6">
            Drag and drop your APK files here, or click to browse
          </p>
          
          <motion.button
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center space-x-2 bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-500 hover:to-blue-500 text-white px-6 py-3 rounded-lg font-medium transition-all duration-200"
            whileHover={{ scale: 1.05, boxShadow: "0 0 20px rgba(0, 255, 136, 0.3)" }}
            whileTap={{ scale: 0.95 }}
          >
            <Upload className="w-5 h-5" />
            <span>Select Files</span>
          </motion.button>
          
          <p className="text-xs text-gray-500 mt-4">
            Supported formats: .apk • Maximum file size: 100MB per file
          </p>
        </div>
      </motion.div>

      {/* Uploaded Files List */}
      <AnimatePresence>
        {uploadedFiles.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-gray-800 border border-gray-700 rounded-xl p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">
                Uploaded Files ({uploadedFiles.length})
              </h3>
              <button
                onClick={() => setUploadedFiles([])}
                className="text-gray-400 hover:text-red-400 transition-colors"
              >
                Clear All
              </button>
            </div>
            
            <div className="space-y-3">
              {uploadedFiles.map((fileObj) => (
                <motion.div
                  key={fileObj.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="flex items-center justify-between p-4 bg-gray-700/50 rounded-lg hover:bg-gray-700 transition-colors"
                >
                  <div className="flex items-center space-x-4">
                    <div className="p-2 bg-gray-600 rounded-lg">
                      <File className="w-5 h-5 text-green-400" />
                    </div>
                    <div>
                      <div className="text-sm font-medium text-white">
                        {fileObj.name}
                      </div>
                      <div className="text-xs text-gray-400">
                        {formatFileSize(fileObj.size)} • Uploaded {fileObj.uploadTime.toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <div className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(fileObj.status)}`}>
                      {fileObj.status.toUpperCase()}
                    </div>
                    
                    {fileObj.status === 'uploaded' && (
                      <motion.button
                        onClick={() => startScan(fileObj)}
                        disabled={isScanning}
                        className="flex items-center space-x-2 bg-green-600 hover:bg-green-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg text-sm font-medium transition-all"
                        whileHover={!isScanning ? { scale: 1.05 } : {}}
                        whileTap={!isScanning ? { scale: 0.95 } : {}}
                      >
                        <Play className="w-4 h-4" />
                        <span>Scan</span>
                      </motion.button>
                    )}
                    
                    {fileObj.status === 'completed' && fileObj.results && (
                      <div className="text-right">
                        <div className={`text-sm font-medium ${
                          fileObj.results.threatLevel === 'critical' ? 'text-red-400' :
                          fileObj.results.threatLevel === 'high' ? 'text-orange-400' :
                          fileObj.results.threatLevel === 'medium' ? 'text-yellow-400' :
                          'text-green-400'
                        }`}>
                          Risk: {fileObj.results.riskScore}%
                        </div>
                        <div className="text-xs text-gray-400">
                          {fileObj.results.status.toUpperCase()}
                        </div>
                      </div>
                    )}
                    
                    <button
                      onClick={() => removeFile(fileObj.id)}
                      className="text-gray-400 hover:text-red-400 transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Scanning Progress */}
      <AnimatePresence>
        {isScanning && scanningFile && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="bg-gray-800 border border-gray-700 rounded-xl p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-yellow-600/20 rounded-lg">
                  <Search className="w-5 h-5 text-yellow-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white">
                    Scanning: {scanningFile.name}
                  </h3>
                  <p className="text-sm text-gray-400">
                    Quantum-level security analysis in progress...
                  </p>
                </div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-yellow-400">
                  {Math.round(scanProgress)}%
                </div>
                <div className="text-sm text-gray-400">Complete</div>
              </div>
            </div>
            
            {/* Progress Bar */}
            <div className="mb-4">
              <div className="flex justify-between text-sm text-gray-400 mb-2">
                <span>{scanPhase}</span>
                <span>{Math.round(scanProgress)}%</span>
              </div>
              <div className="w-full h-3 bg-gray-700 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-green-400 to-blue-400 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${scanProgress}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
            </div>
            
            {/* Scanning Animation */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { icon: Shield, label: 'Security Check', active: scanProgress > 10 },
                { icon: Eye, label: 'Code Analysis', active: scanProgress > 30 },
                { icon: Lock, label: 'Permissions', active: scanProgress > 50 },
                { icon: Activity, label: 'Behavior', active: scanProgress > 70 }
              ].map((item, index) => (
                <motion.div
                  key={index}
                  className={`flex items-center space-x-2 p-3 rounded-lg transition-all ${
                    item.active ? 'bg-green-900/30 text-green-400' : 'bg-gray-700/50 text-gray-500'
                  }`}
                  animate={item.active ? { scale: [1, 1.05, 1] } : {}}
                  transition={{ duration: 0.5, repeat: item.active ? Infinity : 0, repeatDelay: 1 }}
                >
                  <item.icon className="w-5 h-5" />
                  <span className="text-sm font-medium">{item.label}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Quick Actions */}
      {!isScanning && uploadedFiles.length === 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-gray-800/50 border border-gray-700 rounded-xl p-6"
        >
          <h3 className="text-lg font-semibold text-white mb-4">
            Quantum Security Features
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              {
                icon: Zap,
                title: 'Quantum Analysis',
                description: 'Advanced threat detection using quantum algorithms',
                color: 'text-blue-400'
              },
              {
                icon: Shield,
                title: 'Real-time Protection',
                description: 'Continuous monitoring and instant threat response',
                color: 'text-green-400'
              },
              {
                icon: Database,
                title: 'Threat Intelligence',
                description: 'Global database with latest security signatures',
                color: 'text-purple-400'
              }
            ].map((feature, index) => (
              <div key={index} className="text-center">
                <div className={`mx-auto w-12 h-12 rounded-full bg-gray-700/50 flex items-center justify-center mb-3 ${feature.color}`}>
                  <feature.icon className="w-6 h-6" />
                </div>
                <h4 className="text-sm font-semibold text-white mb-1">
                  {feature.title}
                </h4>
                <p className="text-xs text-gray-400">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default APKUpload;
