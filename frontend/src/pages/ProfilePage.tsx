import { motion } from 'framer-motion'
import { UserIcon } from '@heroicons/react/24/outline'

export function ProfilePage() {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-4">
          Profile
        </h1>
        <p className="text-lg text-gray-600 dark:text-gray-300">
          Manage your account and settings
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="card p-8 text-center"
      >
        <UserIcon className="h-16 w-16 text-gray-400 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Profile Management Coming Soon
        </h2>
        <p className="text-gray-600 dark:text-gray-300">
          User profile and settings management will be implemented here
        </p>
      </motion.div>
    </div>
  )
}