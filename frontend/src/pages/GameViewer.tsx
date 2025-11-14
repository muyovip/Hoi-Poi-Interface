import { useParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'

export function GameViewer() {
  const { gameId } = useParams<{ gameId: string }>()
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Simulate loading game data
    const timer = setTimeout(() => {
      setIsLoading(false)
    }, 1000)

    return () => clearTimeout(timer)
  }, [gameId])

  if (isLoading) {
    return (
      <div className="flex justify-center items-center min-h-64">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-4">
          Game Viewer
        </h1>
        <p className="text-lg text-gray-600 dark:text-gray-300">
          Viewing game: {gameId}
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="card p-8 text-center"
      >
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Game Viewer Coming Soon
        </h2>
        <p className="text-gray-600 dark:text-gray-300">
          This page will display the generated game with interactive features
        </p>
      </motion.div>
    </div>
  )
}