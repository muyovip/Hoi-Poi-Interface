import { motion } from 'framer-motion'
import { PlayIcon } from '@heroicons/react/24/outline'

export function PlanetInterface() {
  return (
    <div className="space-y-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-4">
          Planetary Interface
        </h1>
        <p className="text-lg text-gray-600 dark:text-gray-300">
          3D visualization of your game universe
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="h-96 bg-gradient-to-b from-blue-100 to-blue-200 dark:from-blue-900 dark:to-blue-800 rounded-2xl flex items-center justify-center"
      >
        <div className="text-center">
          <PlayIcon className="h-16 w-16 text-blue-600 dark:text-blue-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
            3D Planet Coming Soon
          </h2>
          <p className="text-gray-600 dark:text-gray-300">
            Interactive 3D planetary interface will be implemented here
          </p>
        </div>
      </motion.div>
    </div>
  )
}