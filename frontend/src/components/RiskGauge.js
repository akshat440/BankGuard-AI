import React, { useState, useEffect, useRef } from 'react';
import { motion, useAnimation, useMotionValue, useTransform } from 'framer-motion';
import { 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  XCircle,
  TrendingUp,
  TrendingDown,
  Activity,
  Zap,
  Eye
} from 'lucide-react';

const RiskGauge = ({ 
  riskScore = 0, 
  size = 200, 
  strokeWidth = 12, 
  animated = true,
  showDetails = true,
  title = "Risk Assessment",
  className = ""
}) => {
  const [displayScore, setDisplayScore] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [previousScore, setPreviousScore] = useState(0);
  const controls = useAnimation();
  const gaugeRef = useRef(null);

  // Animation values
  const animatedScore = useMotionValue(0);
  const pathLength = useTransform(animatedScore, [0, 100], [0, 0.75]); // 75% of circle for max score
  const opacity = useTransform(animatedScore, [0, 100], [0.3, 1]);

  // Calculate gauge properties
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const center = size / 2;

  // Determine risk level and colors
  const getRiskLevel = (score) => {
    if (score >= 80) return { level: 'critical', color: '#ef4444', bgColor: '#7f1d1d' };
    if (score >= 60) return { level: 'high', color: '#f97316', bgColor: '#9a3412' };
    if (score >= 40) return { level: 'medium', color: '#eab308', bgColor: '#a16207' };
    if (score >= 20) return { level: 'low', color: '#22c55e', bgColor: '#15803d' };
    return { level: 'minimal', color: '#10b981', bgColor: '#047857' };
  };

  const currentRisk = getRiskLevel(displayScore);
  const targetRisk = getRiskLevel(riskScore);

  // Get risk icon
  const getRiskIcon = (level) => {
    switch (level) {
      case 'critical': return XCircle;
      case 'high': return AlertTriangle;
      case 'medium': return Activity;
      case 'low': return Shield;
      case 'minimal': return CheckCircle;
      default: return Shield;
    }
  };

  const RiskIcon = getRiskIcon(currentRisk.level);

  // Animate score changes
  useEffect(() => {
    if (!animated) {
      setDisplayScore(riskScore);
      animatedScore.set(riskScore);
      return;
    }

    setIsAnimating(true);
    setPreviousScore(displayScore);

    // Animate the score with realistic easing
    const animation = controls.start({
      from: displayScore,
      to: riskScore,
      transition: {
        duration: 2,
        ease: "easeOut",
        type: "tween"
      }
    });

    // Update display score during animation
    let startTime = Date.now();
    const duration = 2000; // 2 seconds
    const startScore = displayScore;
    const scoreChange = riskScore - startScore;

    const updateScore = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easedProgress = 1 - Math.pow(1 - progress, 3); // Ease out cubic
      
      const currentScore = startScore + (scoreChange * easedProgress);
      setDisplayScore(Math.round(currentScore));
      animatedScore.set(currentScore);

      if (progress < 1) {
        requestAnimationFrame(updateScore);
      } else {
        setIsAnimating(false);
      }
    };

    requestAnimationFrame(updateScore);

    return () => {
      animation.stop();
    };
  }, [riskScore, animated, controls, animatedScore, displayScore]);

  // Generate gradient stops for the gauge
  const getGradientStops = () => {
    return [
      { offset: '0%', color: '#10b981' },    // Green - safe
      { offset: '25%', color: '#22c55e' },   // Light green
      { offset: '50%', color: '#eab308' },   // Yellow - medium
      { offset: '75%', color: '#f97316' },   // Orange - high
      { offset: '100%', color: '#ef4444' }   // Red - critical
    ];
  };

  // Calculate the stroke dash array for the progress arc
  const progressLength = (displayScore / 100) * (circumference * 0.75); // 75% max arc
  const remainingLength = (circumference * 0.75) - progressLength;

  return (
    <div className={`relative ${className}`}>
      {/* Title */}
      {title && (
        <motion.h3 
          className="text-lg font-semibold text-white mb-4 text-center"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          {title}
        </motion.h3>
      )}

      {/* Main Gauge Container */}
      <div className="relative inline-block">
        <svg
          ref={gaugeRef}
          width={size}
          height={size}
          className="transform -rotate-90"
          style={{ filter: 'drop-shadow(0 0 10px rgba(0, 0, 0, 0.3))' }}
        >
          {/* Gradient Definition */}
          <defs>
            <linearGradient id="riskGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              {getGradientStops().map((stop, index) => (
                <stop key={index} offset={stop.offset} stopColor={stop.color} />
              ))}
            </linearGradient>
            
            {/* Glow Effect */}
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
              <feMerge> 
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
            
            {/* Pulse Animation */}
            <filter id="pulse">
              <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
              <feMerge> 
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>

          {/* Background Arc */}
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke="#374151"
            strokeWidth={strokeWidth}
            strokeDasharray={`${circumference * 0.75} ${circumference * 0.25}`}
            strokeDashoffset={circumference * 0.125} // Start from bottom left
            opacity={0.3}
          />

          {/* Progress Arc */}
          <motion.circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke={currentRisk.color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={`${progressLength} ${remainingLength}`}
            strokeDashoffset={circumference * 0.125}
            style={{ 
              filter: isAnimating ? 'url(#pulse)' : 'url(#glow)',
              opacity: opacity.get()
            }}
            animate={{
              stroke: targetRisk.color,
            }}
            transition={{ duration: 0.5 }}
          />

          {/* Tick Marks */}
          {[0, 20, 40, 60, 80, 100].map((value, index) => {
            const angle = (value / 100) * 270 - 135; // 270 degrees total, starting from -135
            const tickLength = value % 20 === 0 ? 8 : 4;
            const tickRadius = radius + strokeWidth / 2;
            
            const x1 = center + Math.cos((angle * Math.PI) / 180) * (tickRadius - tickLength);
            const y1 = center + Math.sin((angle * Math.PI) / 180) * (tickRadius - tickLength);
            const x2 = center + Math.cos((angle * Math.PI) / 180) * tickRadius;
            const y2 = center + Math.sin((angle * Math.PI) / 180) * tickRadius;

            return (
              <line
                key={index}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="#6b7280"
                strokeWidth={value % 20 === 0 ? 2 : 1}
                opacity={0.6}
              />
            );
          })}
        </svg>

        {/* Center Content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          {/* Risk Score */}
          <motion.div
            className="text-center"
            animate={{ 
              scale: isAnimating ? [1, 1.1, 1] : 1,
              color: targetRisk.color
            }}
            transition={{ 
              duration: isAnimating ? 0.5 : 0.3,
              repeat: isAnimating ? Infinity : 0 
            }}
          >
            <div className="text-3xl font-bold" style={{ color: currentRisk.color }}>
              {displayScore}
            </div>
            <div className="text-sm text-gray-400 font-medium">
              RISK SCORE
            </div>
          </motion.div>

          {/* Risk Level Indicator */}
          <motion.div
            className="mt-2 flex items-center space-x-1"
            animate={{ opacity: isAnimating ? [0.5, 1, 0.5] : 1 }}
            transition={{ duration: 0.5, repeat: isAnimating ? Infinity : 0 }}
          >
            <RiskIcon 
              className="w-4 h-4" 
              style={{ color: currentRisk.color }}
            />
            <span 
              className="text-xs font-medium uppercase"
              style={{ color: currentRisk.color }}
            >
              {currentRisk.level}
            </span>
          </motion.div>
        </div>

        {/* Animated Particles for High Risk */}
        {displayScore >= 70 && (
          <div className="absolute inset-0 pointer-events-none">
            {[...Array(6)].map((_, i) => (
              <motion.div
                key={i}
                className="absolute w-1 h-1 bg-red-400 rounded-full"
                style={{
                  left: `${45 + Math.random() * 10}%`,
                  top: `${45 + Math.random() * 10}%`
                }}
                animate={{
                  scale: [0, 1, 0],
                  opacity: [0, 1, 0],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  delay: i * 0.3,
                  ease: "easeInOut"
                }}
              />
            ))}
          </div>
        )}
      </div>

      {/* Details Panel */}
      {showDetails && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-6 bg-gray-800/50 rounded-lg p-4 border border-gray-700"
        >
          <div className="grid grid-cols-2 gap-4 text-sm">
            {/* Current Status */}
            <div className="text-center">
              <div className={`text-lg font-bold`} style={{ color: currentRisk.color }}>
                {currentRisk.level.charAt(0).toUpperCase() + currentRisk.level.slice(1)}
              </div>
              <div className="text-gray-400">Threat Level</div>
            </div>

            {/* Score Change */}
            <div className="text-center">
              {previousScore !== displayScore && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="flex items-center justify-center space-x-1"
                >
                  {displayScore > previousScore ? (
                    <>
                      <TrendingUp className="w-4 h-4 text-red-400" />
                      <span className="text-red-400 font-medium">
                        +{displayScore - previousScore}
                      </span>
                    </>
                  ) : displayScore < previousScore ? (
                    <>
                      <TrendingDown className="w-4 h-4 text-green-400" />
                      <span className="text-green-400 font-medium">
                        {displayScore - previousScore}
                      </span>
                    </>
                  ) : (
                    <span className="text-gray-400">No Change</span>
                  )}
                </motion.div>
              )}
              <div className="text-gray-400 text-xs mt-1">Score Change</div>
            </div>
          </div>

          {/* Risk Recommendations */}
          <div className="mt-4 pt-3 border-t border-gray-700">
            <div className="flex items-center space-x-2 mb-2">
              <Eye className="w-4 h-4 text-blue-400" />
              <span className="text-sm font-medium text-white">Recommendation:</span>
            </div>
            <p className="text-xs text-gray-300">
              {displayScore >= 80 
                ? "Critical risk detected. Immediate action required. Do not install this APK."
                : displayScore >= 60
                ? "High risk level. Exercise extreme caution before installation."
                : displayScore >= 40
                ? "Medium risk. Review security findings before proceeding."
                : displayScore >= 20
                ? "Low risk detected. Generally safe with minor concerns."
                : "Minimal risk. This APK appears to be secure for installation."
              }
            </p>
          </div>

          {/* Quantum Shield Status */}
          <div className="mt-3 pt-3 border-t border-gray-700">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Zap className="w-4 h-4 text-blue-400" />
                <span className="text-sm text-white">Quantum Shield</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className={`w-2 h-2 rounded-full ${displayScore < 50 ? 'bg-green-400 animate-pulse' : 'bg-gray-500'}`}></div>
                <span className={`text-xs font-medium ${displayScore < 50 ? 'text-green-400' : 'text-gray-500'}`}>
                  {displayScore < 50 ? 'ACTIVE' : 'STANDBY'}
                </span>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default RiskGauge;
