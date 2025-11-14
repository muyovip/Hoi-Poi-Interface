import { motion } from 'framer-motion'
import { BookOpenIcon, PlusIcon } from '@heroicons/react/24/outline'
import { Link } from 'react-router-dom'

export function GameLibrary() {
  return (
    <div className="space-y-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex justify-between items-center"
      >
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Game Library</h1>
          <p className="text-lg text-gray-600 dark:text-gray-300 mt-2">
            Your collection of AI-generated games
          </p>
        </div>
        <Link
          to="/generate"
          className="btn-primary flex items-center gap-2"
        >
          <PlusIcon className="h-5 w-5" />
          Generate New
        </Link>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="text-center py-16"
      >
        <BookOpenIcon className="h-16 w-16 text-gray-400 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
          No games yet
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          Start by generating your first AI-powered game
        </p>
        <Link
          to="/generate"
          className="btn-primary inline-flex items-center gap-2"
        >
          <PlusIcon className="h-5 w-5" />
          Create Your First Game
        </Link>
      </motion.div>
    </div>
  )
}