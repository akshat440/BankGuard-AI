import React, { useState, useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Shield, 
  Search, 
  BarChart3, 
  FileText, 
  Settings, 
  Bell, 
  User, 
  Menu, 
  X,
  Activity,
  Zap,
  Eye,
  ChevronDown
} from 'lucide-react';

const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [showNotifications, setShowNotifications] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());
  
  const location = useLocation();
  const navigate = useNavigate();

  // Update time every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  // Generate fake notifications
  useEffect(() => {
    const generateNotifications = () => {
      const notificationTypes = [
        { type: 'threat', message: 'High-risk APK detected: malware.apk', severity: 'high' },
        { type: 'scan', message: 'Scan completed for banking_app.apk', severity: 'info' },
        { type: 'shield', message: 'Quantum Shield upgraded to v2.1.4', severity: 'success' },
        { type: 'alert', message: 'Suspicious network activity detected', severity: 'warning' },
        { type: 'system', message: 'Security protocols updated', severity: 'info' }
      ];

      const newNotifications = [];
      for (let i = 0; i < 3; i++) {
        const notification = notificationTypes[Math.floor(Math.random() * notificationTypes.length)];
        newNotifications.push({
          id: Date.now() + i,
          ...notification,
          timestamp: new Date(Date.now() - Math.random() * 3600000) // Random time within last hour
        });
      }
      
      setNotifications(newNotifications);
    };

    generateNotifications();
    const interval = setInterval(generateNotifications, 60000); // Update every minute

    return () => clearInterval(interval);
  }, []);

  // Navigation items
  const navItems = [
    {
      path: '/',
      name: 'Home',
      icon: Shield,
      description: 'Security Overview'
    },
    {
      path: '/dashboard',
      name: 'Dashboard',
      icon: Activity,
      description: 'Real-time Monitoring'
    },
    {
      path: '/analysis',
      name: 'Analysis',
      icon: Search,
      description: 'APK Security Scan'
    },
    {
      path: '/reports',
      name: 'Reports',
      icon: FileText,
      description: 'Security Reports'
    }
  ];

  // User menu items
  const userMenuItems = [
    { name: 'Profile Settings', icon: User, action: () => console.log('Profile') },
    { name: 'Security Settings', icon: Settings, action: () => console.log('Security') },
    { name: 'Quantum Preferences', icon: Zap, action: () => console.log('Quantum') },
    { name: 'Privacy Controls', icon: Eye, action: () => console.log('Privacy') }
  ];

  // Get notification severity color
  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high': return 'text-red-400 bg-red-900/20';
      case 'warning': return 'text-yellow-400 bg-yellow-900/20';
      case 'success': return 'text-green-400 bg-green-900/20';
      default: return 'text-blue-400 bg-blue-900/20';
    }
  };

  // Format time for display
  const formatTime = (date) => {
    return date.toLocaleTimeString('en-US', { 
      hour12: false, 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    });
  };

  // Handle quick scan action
  const handleQuickScan = () => {
    navigate('/analysis');
    // Add visual feedback
    const button = document.querySelector('.quick-scan-btn');
    if (button) {
      button.classList.add('animate-pulse');
      setTimeout(() => {
        button.classList.remove('animate-pulse');
      }, 1000);
    }
  };

  return (
    <>
      {/* Main Navigation Bar */}
      <nav className="fixed top-12 left-0 right-0 bg-gray-800/95 backdrop-blur-md border-b border-gray-700 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            
            {/* Logo and Brand */}
            <motion.div 
              className="flex items-center space-x-4"
              whileHover={{ scale: 1.02 }}
              transition={{ type: 'spring', stiffness: 400, damping: 25 }}
            >
              <Link to="/" className="flex items-center space-x-3">
                <motion.div
                  className="relative"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                >
                  <Shield className="w-8 h-8 text-green-400" />
                  <div className="absolute inset-0 w-8 h-8 border-2 border-green-400/30 rounded-full animate-ping"></div>
                </motion.div>
                <div>
                  <h1 className="text-xl font-bold text-white">Quantum APK</h1>
                  <p className="text-xs text-gray-400 leading-none">Security Analyzer</p>
                </div>
              </Link>
            </motion.div>

            {/* Desktop Navigation */}
            <div className="hidden md:block">
              <div className="flex items-center space-x-1">
                {navItems.map((item) => {
                  const Icon = item.icon;
                  const isActive = location.pathname === item.path;
                  
                  return (
                    <Link
                      key={item.path}
                      to={item.path}
                      className="group relative"
                    >
                      <motion.div
                        className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                          isActive 
                            ? 'bg-green-400/20 text-green-400' 
                            : 'text-gray-300 hover:bg-gray-700/50 hover:text-white'
                        }`}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        <Icon className="w-5 h-5" />
                        <span className="font-medium">{item.name}</span>
                      </motion.div>
                      
                      {/* Active indicator */}
                      {isActive && (
                        <motion.div
                          className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-1 h-1 bg-green-400 rounded-full"
                          layoutId="activeTab"
                          transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                        />
                      )}
                      
                      {/* Hover tooltip */}
                      <div className="absolute top-full mt-2 left-1/2 transform -translate-x-1/2 px-2 py-1 bg-gray-900 text-xs text-white rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                        {item.description}
                      </div>
                    </Link>
                  );
                })}
              </div>
            </div>

            {/* Right Side Actions */}
            <div className="flex items-center space-x-4">
              
              {/* Current Time Display */}
              <div className="hidden sm:block text-right">
                <div className="text-sm font-mono text-green-400">
                  {formatTime(currentTime)}
                </div>
                <div className="text-xs text-gray-400">
                  QUANTUM TIME
                </div>
              </div>

              {/* Quick Scan Button */}
              <motion.button
                onClick={handleQuickScan}
                className="quick-scan-btn hidden sm:flex items-center space-x-2 bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-500 hover:to-blue-500 text-white px-4 py-2 rounded-lg font-medium transition-all duration-200"
                whileHover={{ scale: 1.05, boxShadow: "0 0 20px rgba(0, 255, 136, 0.3)" }}
                whileTap={{ scale: 0.95 }}
              >
                <Search className="w-4 h-4" />
                <span>Quick Scan</span>
              </motion.button>

              {/* Notifications */}
              <div className="relative">
                <motion.button
                  onClick={() => setShowNotifications(!showNotifications)}
                  className="relative p-2 rounded-lg bg-gray-700/50 hover:bg-gray-600/50 transition-colors"
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <Bell className="w-5 h-5 text-gray-300" />
                  {notifications.length > 0 && (
                    <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                  )}
                </motion.button>

                {/* Notifications Dropdown */}
                <AnimatePresence>
                  {showNotifications && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.9, y: -10 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.9, y: -10 }}
                      transition={{ duration: 0.2 }}
                      className="absolute right-0 mt-2 w-80 bg-gray-800 border border-gray-700 rounded-lg shadow-2xl z-50"
                    >
                      <div className="p-4 border-b border-gray-700">
                        <h3 className="text-sm font-semibold text-white">Security Notifications</h3>
                      </div>
                      <div className="max-h-64 overflow-y-auto">
                        {notifications.map((notification) => (
                          <div
                            key={notification.id}
                            className={`p-3 border-b border-gray-700/50 last:border-b-0 ${getSeverityColor(notification.severity)} hover:bg-gray-700/30 transition-colors`}
                          >
                            <p className="text-sm text-white">{notification.message}</p>
                            <p className="text-xs text-gray-400 mt-1">
                              {notification.timestamp.toLocaleTimeString()}
                            </p>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* User Menu */}
              <div className="relative">
                <motion.button
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className="flex items-center space-x-2 p-2 rounded-lg bg-gray-700/50 hover:bg-gray-600/50 transition-colors"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <div className="w-8 h-8 rounded-full bg-gradient-to-r from-green-400 to-blue-400 flex items-center justify-center">
                    <User className="w-4 h-4 text-white" />
                  </div>
                  <ChevronDown className="w-4 h-4 text-gray-400" />
                </motion.button>

                {/* User Dropdown */}
                <AnimatePresence>
                  {showUserMenu && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.9, y: -10 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.9, y: -10 }}
                      transition={{ duration: 0.2 }}
                      className="absolute right-0 mt-2 w-56 bg-gray-800 border border-gray-700 rounded-lg shadow-2xl z-50"
                    >
                      <div className="p-4 border-b border-gray-700">
                        <p className="text-sm font-semibold text-white">Quantum Analyst</p>
                        <p className="text-xs text-gray-400">admin@quantum-security.com</p>
                      </div>
                      <div className="py-2">
                        {userMenuItems.map((item) => {
                          const Icon = item.icon;
                          return (
                            <button
                              key={item.name}
                              onClick={item.action}
                              className="w-full flex items-center space-x-3 px-4 py-2 text-left hover:bg-gray-700/50 transition-colors"
                            >
                              <Icon className="w-4 h-4 text-gray-400" />
                              <span className="text-sm text-white">{item.name}</span>
                            </button>
                          );
                        })}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Mobile Menu Button */}
              <motion.button
                onClick={() => setIsMenuOpen(!isMenuOpen)}
                className="md:hidden p-2 rounded-lg bg-gray-700/50 hover:bg-gray-600/50 transition-colors"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                {isMenuOpen ? (
                  <X className="w-5 h-5 text-gray-300" />
                ) : (
                  <Menu className="w-5 h-5 text-gray-300" />
                )}
              </motion.button>
            </div>
          </div>
        </div>

        {/* Mobile Menu */}
        <AnimatePresence>
          {isMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
              className="md:hidden border-t border-gray-700 bg-gray-800/95 backdrop-blur-md"
            >
              <div className="px-4 py-4 space-y-2">
                {navItems.map((item) => {
                  const Icon = item.icon;
                  const isActive = location.pathname === item.path;
                  
                  return (
                    <Link
                      key={item.path}
                      to={item.path}
                      onClick={() => setIsMenuOpen(false)}
                      className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                        isActive 
                          ? 'bg-green-400/20 text-green-400' 
                          : 'text-gray-300 hover:bg-gray-700/50 hover:text-white'
                      }`}
                    >
                      <Icon className="w-5 h-5" />
                      <div>
                        <div className="font-medium">{item.name}</div>
                        <div className="text-xs text-gray-400">{item.description}</div>
                      </div>
                    </Link>
                  );
                })}
                
                {/* Mobile Quick Scan */}
                <motion.button
                  onClick={() => {
                    handleQuickScan();
                    setIsMenuOpen(false);
                  }}
                  className="w-full flex items-center space-x-3 px-4 py-3 bg-gradient-to-r from-green-600 to-blue-600 text-white rounded-lg font-medium"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <Search className="w-5 h-5" />
                  <span>Quick Scan</span>
                </motion.button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </nav>

      {/* Click outside to close dropdowns */}
      {(showNotifications || showUserMenu) && (
        <div 
          className="fixed inset-0 z-30" 
          onClick={() => {
            setShowNotifications(false);
            setShowUserMenu(false);
          }}
        />
      )}
    </>
  );
};

export default Navbar;
