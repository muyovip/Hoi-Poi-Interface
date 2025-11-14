import { useState } from 'react'
import { motion } from 'framer-motion'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { toast } from 'react-hot-toast'
import {
  SparklesIcon,
  BookOpenIcon,
  CogIcon,
  BeakerIcon,
  ChartBarIcon,
  PlusIcon,
  MinusIcon,
} from '@heroicons/react/24/outline'

import { useCapsule } from '@/hooks/useCapsule'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'

const capsuleSchema = z.object({
  title: z.string().min(1, 'Title is required').max(100, 'Title must be less than 100 characters'),
  description: z.string().min(10, 'Description must be at least 10 characters').max(1000, 'Description must be less than 1000 characters'),
  genre: z.string().optional(),
  complexity: z.enum(['simple', 'medium', 'complex']),
  theme: z.string().optional(),
})

type CapsuleFormData = z.infer<typeof capsuleSchema>

interface CapsuleCreatorProps {
  onCapsuleCreated?: (cid: string) => void
  className?: string
}

export function CapsuleCreator({ onCapsuleCreated, className }: CapsuleCreatorProps) {
  const [step, setStep] = useState(1)
  const [isCreating, setIsCreating] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const { createCapsule, parseGameExpression, buildGameExpression, isCreating: isCreatingCapsule } = useCapsule()

  const {
    register,
    handleSubmit,
    formState: { errors, isDirty },
    watch,
    setValue,
    getValues,
  } = useForm<CapsuleFormData>({
    resolver: zodResolver(capsuleSchema),
    defaultValues: {
      title: '',
      description: '',
      genre: '',
      complexity: 'medium',
      theme: '',
    },
  })

  // Game components state
  const [components, setComponents] = useState({
    narrative: '',
    mechanics: '',
    assets: '',
    balance: '',
  })

  const watchedComplexity = watch('complexity')
  const watchedDescription = watch('description')

  const onSubmit = async (data: CapsuleFormData) => {
    setIsCreating(true)

    try {
      // Build GΛLYPH expression from components
      const gameExpression = buildGameExpression(components)

      // Validate the expression
      const parseResult = await parseGameExpression(gameExpression)
      if (!parseResult.isValid) {
        toast.error(`Invalid game expression: ${parseResult.errors.join(', ')}`)
        return
      }

      // Create the capsule
      const capsule = await createCapsule({
        title: data.title,
        description: data.description,
        genre: data.genre || undefined,
        complexity: data.complexity,
        theme: data.theme || undefined,
        components: components,
      })

      toast.success(`Game "${data.title}" created successfully!`)
      onCapsuleCreated?.(capsule.metadata.game_cid)

      // Reset form
      resetForm()
    } catch (error) {
      console.error('Capsule creation failed:', error)
      toast.error('Failed to create game capsule')
    } finally {
      setIsCreating(false)
    }
  }

  const resetForm = () => {
    setStep(1)
    setComponents({
      narrative: '',
      mechanics: '',
      assets: '',
      balance: '',
    })
  }

  const nextStep = () => {
    if (step < 3) setStep(step + 1)
  }

  const prevStep = () => {
    if (step > 1) setStep(step - 1)
  }

  const updateComponent = (key: keyof typeof components, value: string) => {
    setComponents(prev => ({ ...prev, [key]: value }))
  }

  // Auto-generate components based on description
  const generateComponents = () => {
    const description = watchedDescription
    const complexity = watchedComplexity

    // Simple auto-generation based on complexity
    if (complexity === 'simple') {
      updateComponent('narrative', `Simple story based on: ${description}`)
      updateComponent('mechanics', 'turn_based=true, player_count=1')
      updateComponent('assets', 'sprite_size=16x16, format=png')
      updateComponent('balance', 'difficulty=0.3, progression=linear')
    } else if (complexity === 'medium') {
      updateComponent('narrative', `Engaging story with multiple characters: ${description}`)
      updateComponent('mechanics', 'turn_based=true, player_count=2-4, strategy_depth=medium')
      updateComponent('assets', 'sprite_size=32x32, format=png, animations=basic')
      updateComponent('balance', 'difficulty=0.5, progression=balanced')
    } else {
      updateComponent('narrative', `Complex narrative with branching storylines: ${description}`)
      updateComponent('mechanics', 'turn_based=true, player_count=1-8, strategy_depth=high')
      updateComponent('assets', 'sprite_size=64x64, format=png, animations=advanced')
      updateComponent('balance', 'difficulty=0.7, progression=dynamic')
    }

    toast.success('Game components generated based on your description')
  }

  const steps = [
    { title: 'Basic Info', icon: BookOpenIcon },
    { title: 'Components', icon: CogIcon },
    { title: 'Review', icon: BeakerIcon },
  ]

  return (
    <div className={className}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card p-8"
      >
        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            {steps.map((stepInfo, index) => {
              const isActive = step === index + 1
              const isCompleted = step > index + 1
              const Icon = stepInfo.icon

              return (
                <div key={index} className="flex items-center flex-1">
                  <div
                    className={clsx(
                      'flex items-center justify-center w-10 h-10 rounded-full transition-colors',
                      isActive && 'bg-primary-600 text-white',
                      isCompleted && 'bg-green-600 text-white',
                      !isActive && !isCompleted && 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                    )}
                  >
                    {isCompleted ? (
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    ) : (
                      <Icon className="w-5 h-5" />
                    )}
                  </div>
                  <div className="ml-3 flex-1">
                    <p className={clsx(
                      'text-sm font-medium',
                      isActive && 'text-primary-600 dark:text-primary-400',
                      isCompleted && 'text-green-600 dark:text-green-400',
                      !isActive && !isCompleted && 'text-gray-600 dark:text-gray-400'
                    )}>
                      {stepInfo.title}
                    </p>
                  </div>
                  {index < steps.length - 1 && (
                    <div className={clsx(
                      'flex-1 h-1 mx-4',
                      isCompleted ? 'bg-green-600' : 'bg-gray-200 dark:bg-gray-700'
                    )} />
                  )}
                </div>
              )
            })}
          </div>
        </div>

        {/* Step 1: Basic Information */}
        {step === 1 && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
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
                rows={4}
                className="text-area-field"
                placeholder="Describe your game concept, story, and gameplay..."
              />
              {errors.description && (
                <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.description.message}</p>
              )}
            </div>

            <div className="grid md:grid-cols-3 gap-6">
              <div>
                <label htmlFor="genre" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Genre (Optional)
                </label>
                <select {...register('genre')} id="genre" className="input-field">
                  <option value="">Select a genre</option>
                  <option value="action">Action</option>
                  <option value="adventure">Adventure</option>
                  <option value="rpg">RPG</option>
                  <option value="strategy">Strategy</option>
                  <option value="puzzle">Puzzle</option>
                  <option value="simulation">Simulation</option>
                </select>
              </div>

              <div>
                <label htmlFor="complexity" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Complexity
                </label>
                <select {...register('complexity')} id="complexity" className="input-field">
                  <option value="simple">Simple - Easy to learn</option>
                  <option value="medium">Medium - Balanced gameplay</option>
                  <option value="complex">Complex - Deep mechanics</option>
                </select>
              </div>

              <div>
                <label htmlFor="theme" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Theme (Optional)
                </label>
                <input
                  {...register('theme')}
                  type="text"
                  id="theme"
                  className="input-field"
                  placeholder="e.g., Space, Fantasy, Cyberpunk"
                />
              </div>
            </div>

            <div className="flex justify-end">
              <button
                type="button"
                onClick={nextStep}
                className="btn-primary"
                disabled={!isDirty}
              >
                Next Step
              </button>
            </div>
          </motion.div>
        )}

        {/* Step 2: Game Components */}
        {step === 2 && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Game Components
              </h3>
              <button
                type="button"
                onClick={generateComponents}
                className="btn-outline flex items-center gap-2"
                disabled={!watchedDescription}
              >
                <SparklesIcon className="h-4 w-4" />
                Auto-Generate
              </button>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  <BookOpenIcon className="inline h-4 w-4 mr-1" />
                  Narrative & Story
                </label>
                <textarea
                  value={components.narrative}
                  onChange={(e) => updateComponent('narrative', e.target.value)}
                  rows={3}
                  className="text-area-field"
                  placeholder="Story, characters, lore..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  <CogIcon className="inline h-4 w-4 mr-1" />
                  Game Mechanics
                </label>
                <textarea
                  value={components.mechanics}
                  onChange={(e) => updateComponent('mechanics', e.target.value)}
                  rows={3}
                  className="text-area-field"
                  placeholder="Rules, gameplay mechanics, systems..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  <BeakerIcon className="inline h-4 w-4 mr-1" />
                  Assets & Resources
                </label>
                <textarea
                  value={components.assets}
                  onChange={(e) => updateComponent('assets', e.target.value)}
                  rows={3}
                  className="text-area-field"
                  placeholder="Sprites, sounds, visual assets..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  <ChartBarIcon className="inline h-4 w-4 mr-1" />
                  Balance & Difficulty
                </label>
                <textarea
                  value={components.balance}
                  onChange={(e) => updateComponent('balance', e.target.value)}
                  rows={3}
                  className="text-area-field"
                  placeholder="Difficulty settings, balance parameters..."
                />
              </div>
            </div>

            <div className="flex justify-between">
              <button
                type="button"
                onClick={prevStep}
                className="btn-outline"
              >
                Previous
              </button>
              <button
                type="button"
                onClick={nextStep}
                className="btn-primary"
              >
                Next Step
              </button>
            </div>
          </motion.div>
        )}

        {/* Step 3: Review */}
        {step === 3 && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Review Your Game
            </h3>

            <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6">
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium text-gray-900 dark:text-gray-100">Title:</h4>
                  <p className="text-gray-600 dark:text-gray-300">{watch('title')}</p>
                </div>

                <div>
                  <h4 className="font-medium text-gray-900 dark:text-gray-100">Description:</h4>
                  <p className="text-gray-600 dark:text-gray-300">{watch('description')}</p>
                </div>

                <div>
                  <h4 className="font-medium text-gray-900 dark:text-gray-100">Generated GΛLYPH Expression:</h4>
                  <code className="block bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded p-3 text-sm text-gray-800 dark:text-gray-200">
                    {buildGameExpression(components)}
                  </code>
                </div>
              </div>
            </div>

            <div className="flex justify-between">
              <button
                type="button"
                onClick={prevStep}
                className="btn-outline"
              >
                Previous
              </button>
              <button
                type="button"
                onClick={handleSubmit(onSubmit)}
                disabled={isCreating || isCreatingCapsule}
                className="btn-primary flex items-center gap-2"
              >
                {isCreating || isCreatingCapsule ? (
                  <>
                    <LoadingSpinner size="sm" />
                    Creating...
                  </>
                ) : (
                  <>
                    <SparklesIcon className="h-5 w-5" />
                    Create Game
                  </>
                )}
              </button>
            </div>
          </motion.div>
        )}
      </motion.div>
    </div>
  )
}