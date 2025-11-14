import { useState } from 'react'
import { motion } from 'framer-motion'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { toast } from 'react-hot-toast'
import { SparklesIcon, PlayIcon } from '@heroicons/react/24/outline'

import { LoadingSpinner } from '@/components/ui/LoadingSpinner'

const gameSchema = z.object({
  title: z.string().min(1, 'Title is required').max(100, 'Title must be less than 100 characters'),
  description: z.string().min(10, 'Description must be at least 10 characters').max(1000, 'Description must be less than 1000 characters'),
  genre: z.string().optional(),
  complexity: z.enum(['simple', 'medium', 'complex'], {
    required_error: 'Please select a complexity level',
  }),
  theme: z.string().optional(),
})

type GameFormData = z.infer<typeof gameSchema>

export function GameGenerator() {
  const [isGenerating, setIsGenerating] = useState(false)
  const [generationProgress, setGenerationProgress] = useState(0)

  const {
    register,
    handleSubmit,
    formState: { errors, isDirty },
    watch,
    reset,
  } = useForm<GameFormData>({
    resolver: zodResolver(gameSchema),
    defaultValues: {
      title: '',
      description: '',
      genre: '',
      complexity: 'medium',
      theme: '',
    },
  })

  const watchedDescription = watch('description')

  const onSubmit = async (data: GameFormData) => {
    setIsGenerating(true)
    setGenerationProgress(0)

    try {
      // Simulate generation progress
      const progressInterval = setInterval(() => {
        setGenerationProgress(prev => Math.min(prev + 10, 90))
      }, 1000)

      // TODO: Replace with actual API call
      await new Promise(resolve => setTimeout(resolve, 12000)) // 12 seconds

      clearInterval(progressInterval)
      setGenerationProgress(100)

      toast.success('Game generated successfully!')

      // TODO: Navigate to game viewer
      reset()
    } catch (error) {
      console.error('Generation error:', error)
      toast.error('Failed to generate game. Please try again.')
    } finally {
      setIsGenerating(false)
      setGenerationProgress(0)
    }
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
          Describe your dream game and let our AI bring it to life
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="card p-8"
      >
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          <div>
            <label htmlFor="title" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Game Title
            </label>
            <input
              {...register('title')}
              type="text"
              id="title"
              className="input-field"
              placeholder="Enter your game title"
              disabled={isGenerating}
            />
            {errors.title && (
              <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.title.message}</p>
            )}
          </div>

          <div>
            <label htmlFor="description" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Game Description
            </label>
            <textarea
              {...register('description')}
              id="description"
              rows={6}
              className="text-area-field"
              placeholder="Describe your game concept, story, mechanics, and what makes it unique..."
              disabled={isGenerating}
            />
            {errors.description && (
              <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.description.message}</p>
            )}
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              {watchedDescription?.length || 0}/1000 characters
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <label htmlFor="genre" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Genre (Optional)
              </label>
              <select
                {...register('genre')}
                id="genre"
                className="input-field"
                disabled={isGenerating}
              >
                <option value="">Select a genre</option>
                <option value="action">Action</option>
                <option value="adventure">Adventure</option>
                <option value="rpg">RPG</option>
                <option value="strategy">Strategy</option>
                <option value="puzzle">Puzzle</option>
                <option value="simulation">Simulation</option>
                <option value="sports">Sports</option>
                <option value="racing">Racing</option>
              </select>
            </div>

            <div>
              <label htmlFor="complexity" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Complexity
              </label>
              <select
                {...register('complexity')}
                id="complexity"
                className="input-field"
                disabled={isGenerating}
              >
                <option value="simple">Simple - Quick to learn</option>
                <option value="medium">Medium - Balanced gameplay</option>
                <option value="complex">Complex - Deep mechanics</option>
              </select>
              {errors.complexity && (
                <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.complexity.message}</p>
              )}
            </div>
          </div>

          <div>
            <label htmlFor="theme" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Theme/Setting (Optional)
            </label>
            <input
              {...register('theme')}
              type="text"
              id="theme"
              className="input-field"
              placeholder="e.g., Space exploration, Fantasy kingdom, Cyberpunk city"
              disabled={isGenerating}
            />
          </div>

          {isGenerating && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="space-y-4"
            >
              <div className="bg-primary-50 dark:bg-primary-900/20 rounded-lg p-4">
                <div className="flex items-center gap-3 mb-2">
                  <LoadingSpinner size="sm" />
                  <span className="text-sm font-medium text-primary-700 dark:text-primary-300">
                    Generating your game...
                  </span>
                </div>
                <div className="w-full bg-primary-200 dark:bg-primary-800 rounded-full h-2">
                  <motion.div
                    className="bg-primary-600 h-2 rounded-full"
                    style={{ width: `${generationProgress}%` }}
                    initial={{ width: 0 }}
                    animate={{ width: `${generationProgress}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
                <p className="text-xs text-primary-600 dark:text-primary-400 mt-2">
                  Coordinating 4 AI models to create your unique game experience
                </p>
              </div>
            </motion.div>
          )}

          <div className="flex justify-center">
            <button
              type="submit"
              disabled={isGenerating || !isDirty}
              className="btn-primary flex items-center gap-2 px-8 py-3 text-lg disabled:opacity-50"
            >
              {isGenerating ? (
                <>
                  <LoadingSpinner size="sm" />
                  Generating...
                </>
              ) : (
                <>
                  <SparklesIcon className="h-5 w-5" />
                  Generate Game
                </>
              )}
            </button>
          </div>
        </form>
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
          { name: 'TinyLlama', role: 'Assets', description: 'Generates visual and audio assets' },
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