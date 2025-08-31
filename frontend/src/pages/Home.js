import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { 
  Shield, 
  Zap, 
  Eye, 
  Lock, 
  Activity, 
  Target, 
  Database, 
  Globe, 
  Users, 
  TrendingUp,
  CheckCircle,
  AlertTriangle,
  Play,
  ArrowRight,
  Star,
  Award,
  Cpu,
  Network,
  Search,
  FileText,
  BarChart3
} from 'lucide-react';

const Home = ({ globalStats, threatLevel }) => {
  const [currentTestimonial, setCurrentTestimonial] = useState(0);
  const [liveStats, setLiveStats] = useState({
    scansToday: 15847,
    threatsBlocked: 2934,
    quantumDetections: 847
  });
  const [featuredThreats, setFeaturedThreats] = useState([]);
  
  const navigate = useNavigate();

  // Initialize featured threats
  useEffect(() => {
    const threats = [
      {
        name: 'QUANTUM_BANKING_TROJAN',
        severity: 'critical',
        detections: 1247,
        trend: '+23%',
        description: 'Advanced banking malware with quantum evasion'
      },
      {
        name: 'NEURAL_SPYWARE_V4',
        severity: 'high', 
        detections: 892,
        trend: '+15%',
        description: 'AI-powered surveillance malware'
      },
      {
        name: 'CRYPTO_MINING_BOTNET',
        severity: 'medium',
        detections: 2156,
        trend: '+8%',
        description: 'Distributed cryptocurrency mining'
      }
    ];
    setFeaturedThreats(threats);
  }, []);

  // Update live stats
  useEffect(() => {
    const interval = setInterval(() => {
      setLiveStats(prev => ({
        scansToday: prev.scansToday + Math.floor(Math.random() * 5) + 1,
        threatsBlocked: prev.threatsBlocked + Math.floor(Math.random() * 2),
        quantumDetections: prev.quantumDetections + Math.floor(Math.random() * 3)
      }));
    }, 10000); // Update every 10 seconds

    return () => clearInterval(interval);
  }, []);

  // Rotate testimonials
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTestimonial(prev => (prev + 1) % testimonials.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const testimonials = [
    {
      name: "Dr. Sarah Chen",
      role: "Chief Security Officer, TechCorp",
      content: "Quantum APK Analyzer detected threats that other solutions missed. The quantum-level analysis is revolutionary.",
      rating: 5
    },
    {
      name: "Marcus Rodriguez", 
      role: "Mobile Security Researcher",
      content: "The most comprehensive APK analysis platform I've used. The evidence reporting is incredibly detailed.",
      rating: 5
    },
    {
      name: "Emily Watson",
      role: "IT Security Manager, FinanceGlobal", 
      content: "Zero false positives in 6 months of use. The quantum threat detection is truly next-generation.",
      rating: 5
    }
  ];

  const features = [
    {
      icon: Zap,
      title: 'Quantum Analysis Engine',
      description: 'Revolutionary quantum-powered threat detection using advanced cryptographic analysis',
      color: 'text-blue-400',
      bgColor: 'bg-blue-900/20'
    },
    {
      icon: Shield,
      title: 'Real-time Protection',
      description: 'Continuous monitoring with instant threat response and automated containment',
      color: 'text-green-400',
      bgColor: 'bg-green-900/20'
    },
    {
      icon: Eye,
      title: 'Deep Code Inspection',
      description: 'Advanced static and dynamic analysis of APK bytecode and runtime behavior',
      color: 'text-purple-400',
      bgColor: 'bg-purple-900/20'
    },
    {
      icon: Database,
      title: 'Global Threat Intelligence',
      description: 'Access to worldwide threat database with ML-powered pattern recognition',
      color: 'text-orange-400',
      bgColor: 'bg-orange-900/20'
    },
    {
      icon: Lock,
      title: 'Zero-Day Detection',
      description: 'Identify previously unknown threats using behavioral analysis and heuristics',
      color: 'text-red-400',
      bgColor: 'bg-red-900/20'
    },
    {
      icon: Network,
      title: 'Network Forensics',
      description: 'Analyze network communications and identify malicious command & control servers',
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-900/20'
    }
  ];

  const stats = [
    {
      value: globalStats?.appsProtected?.toLocaleString() || '12,847',
      label: 'Apps Protected',
      icon: Shield,
      color: 'text-green-400',
      change: '+12%'
    },
    {
      value: liveStats.scansToday.toLocaleString(),
      label: 'Scans Today',
      icon: Search,
      color: 'text-blue-400',
      change: '+8%'
    },
    {
      value: liveStats.threatsBlocked.toLocaleString(),
      label: 'Threats Blocked',
      icon: Target,
      color: 'text-red-400',
      change: '+15%'
    },
    {
      value: liveStats.quantumDetections.toLocaleString(),
      label: 'Quantum Detections',
      icon: Zap,
      color: 'text-purple-400',
      change: '+23%'
    }
  ];

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical': return 'text-red-400';
      case 'high': return 'text-orange-400';
      case 'medium': return 'text-yellow-400';
      default: return 'text-green-400';
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      
      {/* Hero Section */}
      <motion.section
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="relative py-20 px-6 overflow-hidden"
      >
        {/* Background Effect */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-900/20 via-gray-900 to-green-900/20"></div>
        
        <div className="relative max-w-7xl mx-auto">
          <div className="text-center">
            <motion.div
              initial={{ scale: 0, rotate: -180 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ duration: 1, type: "spring", stiffness: 100 }}
              className="mx-auto w-24 h-24 mb-8 relative"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-green-400 to-blue-500 rounded-full animate-pulse"></div>
              <div className="absolute inset-2 bg-gray-900 rounded-full flex items-center justify-center">
                <Shield className="w-10 h-10 text-green-400" />
              </div>
            </motion.div>
            
            <motion.h1
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-green-400 via-blue-400 to-purple-400 bg-clip-text text-transparent"
            >
              Quantum APK
              <br />
              Security Analyzer
            </motion.h1>
            
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto"
            >
              Next-generation APK security analysis powered by quantum computing. 
              Detect advanced threats, zero-day exploits, and sophisticated malware with unprecedented accuracy.
            </motion.p>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
              className="flex flex-col sm:flex-row items-center justify-center gap-4"
            >
              <motion.button
                onClick={() => navigate('/analysis')}
                className="group bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-500 hover:to-blue-500 text-white px-8 py-4 rounded-xl text-lg font-semibold transition-all duration-300 flex items-center space-x-3"
                whileHover={{ scale: 1.05, boxShadow: "0 20px 40px rgba(0, 255, 136, 0.3)" }}
                whileTap={{ scale: 0.95 }}
              >
                <Play className="w-5 h-5" />
                <span>Start Analysis</span>
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </motion.button>
              
              <motion.button
                onClick={() => navigate('/dashboard')}
                className="group border border-gray-600 hover:border-gray-400 text-gray-300 hover:text-white px-8 py-4 rounded-xl text-lg font-semibold transition-all duration-300 flex items-center space-x-3"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <BarChart3 className="w-5 h-5" />
                <span>View Dashboard</span>
              </motion.button>
            </motion.div>
          </div>
        </div>
      </motion.section>

      {/* Live Stats Section */}
      <motion.section
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9 }}
        className="py-16 px-6 border-t border-gray-800"
      >
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <motion.h2
              className="text-3xl font-bold text-white mb-4"
              animate={{ opacity: [0.7, 1, 0.7] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              Live Security Statistics
            </motion.h2>
            <p className="text-gray-400">Real-time threat detection and protection metrics</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {stats.map((stat, index) => {
              const Icon = stat.icon;
              return (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 1 + index * 0.1 }}
                  className="bg-gray-800 border border-gray-700 rounded-xl p-6 hover:bg-gray-700/50 transition-all duration-300 group"
                  whileHover={{ scale: 1.05, boxShadow: "0 10px 30px rgba(0, 0, 0, 0.3)" }}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className={`p-3 rounded-lg bg-gray-700/50 ${stat.color} group-hover:scale-110 transition-transform`}>
                      <Icon className="w-6 h-6" />
                    </div>
                    <div className="text-right">
                      <div className="text-xs text-green-400 font-medium">{stat.change}</div>
                    </div>
                  </div>
                  <div className="text-2xl font-bold text-white mb-1">{stat.value}</div>
                  <div className="text-sm text-gray-400">{stat.label}</div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </motion.section>

      {/* Features Section */}
      <motion.section
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        className="py-20 px-6 bg-gray-800/30"
      >
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="text-4xl font-bold text-white mb-6"
            >
              Advanced Security Features
            </motion.h2>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              className="text-xl text-gray-400 max-w-3xl mx-auto"
            >
              Powered by quantum computing and machine learning, our platform provides 
              unparalleled threat detection capabilities for modern mobile security challenges.
            </motion.p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  className="group"
                >
                  <motion.div
                    className="bg-gray-800 border border-gray-700 rounded-xl p-8 h-full hover:border-gray-600 transition-all duration-300"
                    whileHover={{ scale: 1.02, y: -5 }}
                  >
                    <div className={`inline-flex items-center justify-center w-16 h-16 rounded-xl ${feature.bgColor} mb-6 group-hover:scale-110 transition-transform`}>
                      <Icon className={`w-8 h-8 ${feature.color}`} />
                    </div>
                    <h3 className="text-xl font-semibold text-white mb-4">{feature.title}</h3>
                    <p className="text-gray-400 leading-relaxed">{feature.description}</p>
                  </motion.div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </motion.section>

      {/* Featured Threats Section */}
      <motion.section
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        className="py-20 px-6"
      >
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="text-3xl font-bold text-white mb-4"
            >
              Latest Threat Intelligence
            </motion.h2>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              className="text-gray-400"
            >
              Recent threats detected by our quantum analysis engine
            </motion.p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {featuredThreats.map((threat, index) => (
              <motion.div
                key={threat.name}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="bg-gray-800 border border-gray-700 rounded-xl p-6 hover:bg-gray-700/50 transition-all duration-300"
                whileHover={{ scale: 1.03 }}
              >
                <div className="flex items-center justify-between mb-4">
                  <div className={`px-3 py-1 rounded-full text-xs font-medium ${getSeverityColor(threat.severity)} bg-gray-700/50`}>
                    {threat.severity.toUpperCase()}
                  </div>
                  <div className="flex items-center space-x-1">
                    <TrendingUp className="w-4 h-4 text-green-400" />
                    <span className="text-sm text-green-400 font-medium">{threat.trend}</span>
                  </div>
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">{threat.name}</h3>
                <p className="text-gray-400 text-sm mb-4">{threat.description}</p>
                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-500">
                    <span className="font-medium text-white">{threat.detections.toLocaleString()}</span> detections
                  </div>
                  <AlertTriangle className={`w-5 h-5 ${getSeverityColor(threat.severity)}`} />
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.section>

      {/* Testimonials Section */}
      <motion.section
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        className="py-20 px-6 bg-gray-800/30"
      >
        <div className="max-w-4xl mx-auto text-center">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-3xl font-bold text-white mb-12"
          >
            Trusted by Security Professionals
          </motion.h2>
          
          <AnimatePresence mode="wait">
            <motion.div
              key={currentTestimonial}
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -50 }}
              transition={{ duration: 0.5 }}
              className="bg-gray-800 border border-gray-700 rounded-xl p-8 max-w-3xl mx-auto"
            >
              <div className="flex justify-center mb-4">
                {[...Array(testimonials[currentTestimonial].rating)].map((_, i) => (
                  <Star key={i} className="w-5 h-5 text-yellow-400 fill-current" />
                ))}
              </div>
              <blockquote className="text-xl text-gray-300 mb-6 italic">
                "{testimonials[currentTestimonial].content}"
              </blockquote>
              <div>
                <div className="font-semibold text-white">
                  {testimonials[currentTestimonial].name}
                </div>
                <div className="text-sm text-gray-400">
                  {testimonials[currentTestimonial].role}
                </div>
              </div>
            </motion.div>
          </AnimatePresence>
          
          <div className="flex justify-center mt-6 space-x-2">
            {testimonials.map((_, index) => (
              <button
                key={index}
                onClick={() => setCurrentTestimonial(index)}
                className={`w-3 h-3 rounded-full transition-all ${
                  index === currentTestimonial ? 'bg-green-400' : 'bg-gray-600'
                }`}
              />
            ))}
          </div>
        </div>
      </motion.section>

      {/* CTA Section */}
      <motion.section
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        className="py-20 px-6"
      >
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="bg-gradient-to-r from-green-900/50 to-blue-900/50 border border-gray-700 rounded-2xl p-12"
          >
            <div className="flex justify-center mb-6">
              <div className="p-4 bg-gradient-to-r from-green-400 to-blue-500 rounded-full">
                <Award className="w-12 h-12 text-white" />
              </div>
            </div>
            <h2 className="text-4xl font-bold text-white mb-6">
              Ready to Secure Your APKs?
            </h2>
            <p className="text-xl text-gray-300 mb-8">
              Join thousands of security professionals who trust Quantum APK Security Analyzer 
              to protect their mobile applications from advanced threats.
            </p>
            <motion.button
              onClick={() => navigate('/analysis')}
              className="bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-500 hover:to-blue-500 text-white px-10 py-4 rounded-xl text-xl font-semibold transition-all duration-300 inline-flex items-center space-x-3"
              whileHover={{ scale: 1.05, boxShadow: "0 20px 40px rgba(0, 255, 136, 0.3)" }}
              whileTap={{ scale: 0.95 }}
            >
              <Shield className="w-6 h-6" />
              <span>Start Free Analysis</span>
            </motion.button>
          </motion.div>
        </div>
      </motion.section>

      {/* Floating Action Button */}
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ delay: 2, type: "spring", stiffness: 200 }}
        className="fixed bottom-8 right-8 z-50"
      >
        <motion.button
          onClick={() => navigate('/analysis')}
          className="bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-500 hover:to-blue-500 text-white p-4 rounded-full shadow-2xl transition-all duration-300"
          whileHover={{ scale: 1.1, boxShadow: "0 10px 30px rgba(0, 255, 136, 0.4)" }}
          whileTap={{ scale: 0.9 }}
          animate={{ 
            y: [0, -10, 0],
            boxShadow: [
              "0 5px 20px rgba(0, 255, 136, 0.3)",
              "0 10px 40px rgba(0, 255, 136, 0.5)",
              "0 5px 20px rgba(0, 255, 136, 0.3)"
            ]
          }}
          transition={{ 
            duration: 2, 
            repeat: Infinity,
            ease: "easeInOut"
          }}
        >
          <Search className="w-6 h-6" />
        </motion.button>
      </motion.div>
    </div>
  );
};

export default Home;
