/**
 * 3D Planetary Interface Component
 *
 * Interactive 3D visualization of the game universe using Three.js and React Three Fiber.
 * Displays games as planets orbiting around a central star, with interactive features
 * for exploring and managing the game collection.
 */

import { Suspense, useRef, useState, useEffect, useMemo } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Stars, Text, Float } from '@react-three/drei'
import { motion } from 'framer-motion'
import * as THREE from 'three'

import { useCapsulesList } from '@/hooks/useCapsule'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { clsx } from 'clsx'

// ============================================================================
// Planet Component
// ============================================================================

interface PlanetProps {
  game: any
  position: [number, number, number]
  size: number
  color: string
  onClick: (game: any) => void
  isSelected: boolean
}

function Planet({ game, position, size, color, onClick, isSelected }: PlanetProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const [hovered, setHovered] = useState(false)

  useFrame((state) => {
    if (meshRef.current) {
      // Orbital motion
      const time = state.clock.getElapsedTime()
      meshRef.current.position.x = position[0] + Math.cos(time * 0.5) * 2
      meshRef.current.position.z = position[2] + Math.sin(time * 0.5) * 2
      meshRef.current.rotation.y += 0.01

      // Scale up when hovered or selected
      const scale = isSelected ? 1.3 : hovered ? 1.2 : 1
      meshRef.current.scale.setScalar(scale)
    }
  })

  return (
    <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
      <mesh
        ref={meshRef}
        position={position}
        onClick={(e) => {
          e.stopPropagation()
          onClick(game)
        }}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <sphereGeometry args={[size, 32, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={isSelected ? color : hovered ? color : '#000000'}
          emissiveIntensity={isSelected ? 0.3 : hovered ? 0.1 : 0}
          metalness={0.4}
          roughness={0.3}
        />
      </mesh>

      {/* Game title label */}
      {(hovered || isSelected) && (
        <Text
          position={[0, size + 0.5, 0]}
          fontSize={0.3}
          color="white"
          anchorX="center"
          anchorY="middle"
        >
          {game.title}
        </Text>
      )}
    </Float>
  )
}

// ============================================================================
// Central Star Component
// ============================================================================

function CentralStar() {
  const meshRef = useRef<THREE.Mesh>(null)

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.005
      const scale = 1 + Math.sin(state.clock.getElapsedTime() * 2) * 0.1
      meshRef.current.scale.setScalar(scale)
    }
  })

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[1, 32, 32]} />
      <meshStandardMaterial
        color="#ffaa00"
        emissive="#ff6600"
        emissiveIntensity={0.8}
        metalness={0.1}
        roughness={0.2}
      />
    </mesh>
  )
}

// ============================================================================
// Orbit Rings Component
// ============================================================================

interface OrbitRingProps {
  radius: number
  rotation: number
}

function OrbitRing({ radius, rotation }: OrbitRingProps) {
  const points = useMemo(() => {
    const pts = []
    for (let i = 0; i <= 64; i++) {
      const angle = (i / 64) * Math.PI * 2
      pts.push(new THREE.Vector3(
        Math.cos(angle) * radius,
        0,
        Math.sin(angle) * radius
      ))
    }
    return pts
  }, [radius])

  return (
    <line>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={points.length}
          array={new Float32Array(points.flatMap(p => [p.x, p.y, p.z]))}
          itemSize={3}
        />
      </bufferGeometry>
      <lineBasicMaterial
        color="#4a5568"
        opacity={0.3}
        transparent
        rotation={[0, rotation, 0]}
      />
    </line>
  )
}

// ============================================================================
// Scene Component
// ============================================================================

interface SceneProps {
  onPlanetClick: (game: any) => void
  selectedGame: any
}

function Scene({ onPlanetClick, selectedGame }: SceneProps) {
  const { data: capsules = [], isLoading } = useCapsulesList()

  if (isLoading) {
    return null
  }

  // Generate planet positions and colors
  const planets = capsules.map((game, index) => {
    const angle = (index / capsules.length) * Math.PI * 2
    const radius = 3 + (index % 3) * 2 // Different orbital radii
    const height = (index % 2) * 1.5 - 0.75 // Different heights

    const colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']
    const color = colors[index % colors.length]

    return {
      game,
      position: [
        Math.cos(angle) * radius,
        height,
        Math.sin(angle) * radius
      ] as [number, number, number],
      size: 0.2 + Math.random() * 0.3,
      color,
    }
  })

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <pointLight position={[-10, -10, -10]} intensity={0.4} color="#4a90e2" />

      {/* Central Star */}
      <CentralStar />

      {/* Orbit Rings */}
      {[3, 5, 7].map((radius, index) => (
        <OrbitRing
          key={radius}
          radius={radius}
          rotation={index * 0.1}
        />
      ))}

      {/* Planets */}
      {planets.map((planet, index) => (
        <Planet
          key={index}
          game={planet.game}
          position={planet.position}
          size={planet.size}
          color={planet.color}
          onClick={onPlanetClick}
          isSelected={selectedGame?.metadata.game_cid === planet.game.metadata.game_cid}
        />
      ))}

      {/* Starfield Background */}
      <Stars
        radius={100}
        depth={50}
        count={5000}
        factor={4}
        saturation={0}
        fade
        speed={1}
      />
    </>
  )
}

// ============================================================================
// Main Planet Interface Component
// ============================================================================

interface PlanetInterfaceProps {
  className?: string
}

export function PlanetInterface({ className }: PlanetInterfaceProps) {
  const [selectedGame, setSelectedGame] = useState<any>(null)
  const [showInfo, setShowInfo] = useState(false)

  const handlePlanetClick = (game: any) => {
    setSelectedGame(game)
    setShowInfo(true)
  }

  const handleCloseInfo = () => {
    setShowInfo(false)
    setTimeout(() => setSelectedGame(null), 300)
  }

  return (
    <div className={clsx('relative w-full h-screen bg-black', className)}>
      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [8, 6, 8], fov: 45 }}
        className="w-full h-full"
      >
        <Suspense fallback={null}>
          <Scene
            onPlanetClick={handlePlanetClick}
            selectedGame={selectedGame}
          />
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            minDistance={3}
            maxDistance={20}
            autoRotate={true}
            autoRotateSpeed={0.5}
          />
        </Suspense>
      </Canvas>

      {/* UI Overlay */}
      <div className="absolute top-0 left-0 right-0 p-6 pointer-events-none">
        <div className="flex justify-between items-start">
          {/* Title */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="pointer-events-auto"
          >
            <h1 className="text-3xl font-bold text-white mb-2">
              Game Universe
            </h1>
            <p className="text-gray-300">
              Navigate your collection of generated games
            </p>
          </motion.div>

          {/* Controls Help */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-black/50 backdrop-blur-sm rounded-lg p-4 pointer-events-auto"
          >
            <h3 className="text-white font-semibold mb-2">Controls</h3>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>🖱️ Left click + drag to rotate</li>
              <li>🔍 Scroll to zoom in/out</li>
              <li>🎯 Click planets to view details</li>
              <li>🌍 Auto-rotation enabled</li>
            </ul>
          </motion.div>
        </div>
      </div>

      {/* Game Info Panel */}
      {selectedGame && (
        <motion.div
          initial={{ opacity: 0, x: 300 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 300 }}
          className="absolute right-0 top-0 h-full w-96 bg-black/80 backdrop-blur-md p-6 pointer-events-auto overflow-y-auto"
        >
          <div className="flex justify-between items-start mb-4">
            <h2 className="text-xl font-bold text-white">
              {selectedGame.manifest.title}
            </h2>
            <button
              onClick={handleCloseInfo}
              className="text-gray-400 hover:text-white transition-colors"
            >
              ✕
            </button>
          </div>

          <div className="space-y-4 text-gray-300">
            <div>
              <h3 className="font-semibold text-white mb-1">Description</h3>
              <p className="text-sm">{selectedGame.manifest.story}</p>
            </div>

            {selectedGame.manifest.genre && (
              <div>
                <h3 className="font-semibold text-white mb-1">Genre</h3>
                <span className="inline-block px-2 py-1 bg-primary-600 text-white text-xs rounded-full">
                  {selectedGame.manifest.genre}
                </span>
              </div>
            )}

            <div>
              <h3 className="font-semibold text-white mb-1">Balance Score</h3>
              <div className="flex items-center gap-2">
                <div className="flex-1 bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-green-500 h-2 rounded-full"
                    style={{ width: `${selectedGame.manifest.balance * 100}%` }}
                  />
                </div>
                <span className="text-sm">
                  {(selectedGame.manifest.balance * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            <div>
              <h3 className="font-semibold text-white mb-1">Created</h3>
              <p className="text-sm">
                {new Date(selectedGame.metadata.created_at).toLocaleDateString()}
              </p>
            </div>

            <div>
              <h3 className="font-semibold text-white mb-1">CID</h3>
              <p className="text-xs font-mono text-gray-400">
                {selectedGame.metadata.game_cid}
              </p>
            </div>

            <div className="pt-4">
              <a
                href={`/games/${selectedGame.metadata.game_cid}`}
                className="btn-primary w-full text-center"
              >
                View Game Details
              </a>
            </div>
          </div>
        </motion.div>
      )}

      {/* Loading State */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="bg-black/50 backdrop-blur-sm rounded-lg p-4">
          <LoadingSpinner size="lg" />
        </div>
      </div>

      {/* Bottom Stats */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="absolute bottom-0 left-0 right-0 p-6 pointer-events-none"
      >
        <div className="bg-black/50 backdrop-blur-sm rounded-lg p-4 pointer-events-auto">
          <div className="flex justify-around text-center">
            <div>
              <p className="text-2xl font-bold text-white">0</p>
              <p className="text-sm text-gray-300">Total Games</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-green-400">0</p>
              <p className="text-sm text-gray-300">Completed</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-yellow-400">0</p>
              <p className="text-sm text-gray-300">In Progress</p>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}