import { motion } from 'framer-motion'

export function Logo() {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      className="flex items-center gap-2"
    >
      <div className="relative">
        <div className="h-8 w-8 rounded-full bg-gradient-to-r from-primary-500 to-accent-500 p-0.5">
          <div className="h-full w-full rounded-full bg-white dark:bg-gray-900 flex items-center justify-center">
            <span className="text-xs font-bold bg-gradient-to-r from-primary-600 to-accent-600 bg-clip-text text-transparent">
              ⊙₀
            </span>
          </div>
        </div>
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          className="absolute -inset-1 rounded-full border border-dashed border-primary-300 dark:border-primary-700 opacity-30"
        />
      </div>
      <span className="text-lg font-bold text-gray-900 dark:text-gray-100">
        CapsuleOS
      </span>
    </motion.div>
  )
}