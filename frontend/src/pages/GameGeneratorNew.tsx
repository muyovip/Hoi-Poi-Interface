import { useState } from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { toast } from 'react-hot-toast'
import { SparklesIcon } from '@heroicons/react/24/outline'

import { CapsuleCreator } from '@/components/capsule/CapsuleCreator'

export function GameGeneratorNew() {
  const navigate = useNavigate()

  const handleCapsuleCreated = (cid: string) => {
    toast.success('Game created successfully! Redirecting to your game...')
    setTimeout(() => {
      navigate(`/games/${cid}`)
    }, 2000)
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-4">
          Generate Your Game
        </h1>
        <p className="text-lg text-gray-600 dark:text-gray-300">
          Create unique games using our multi-LLM orchestration system
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <CapsuleCreator onCapsuleCreated={handleCapsuleCreated} />
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="grid md:grid-cols-4 gap-4"
      >
        {[
          { name: 'Phi-3', role: 'Narrative', description: 'Crafts compelling stories and characters' },
          { name: 'Gemma-2B', role: 'Mechanics', description: 'Designs balanced gameplay systems' },
          { name: 'TinyLlama', role: 'Assets', description: 'Generates detailed asset specifications' },
          { name: 'Qwen-0.5B', role: 'Balance', description: 'Optimizes game difficulty and pacing' },
        ].map((model, index) => (
          <motion.div
            key={model.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 + index * 0.1 }}
            className="text-center p-4"
          >
            <div className="w-12 h-12 bg-gradient-to-r from-primary-500 to-accent-500 rounded-full flex items-center justify-center mx-auto mb-2">
              <span className="text-white font-bold text-sm">{model.name[0]}</span>
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-gray-100">{model.name}</h3>
            <p className="text-xs text-primary-600 dark:text-primary-400 mb-1">{model.role}</p>
            <p className="text-xs text-gray-600 dark:text-gray-400">{model.description}</p>
          </motion.div>
        ))}
      </motion.div>
    </div>
  )
}