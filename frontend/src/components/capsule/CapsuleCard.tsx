import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { formatDistanceToNow } from 'date-fns'
import {
  BookOpenIcon,
  SparklesIcon,
  CogIcon,
  ChartBarIcon,
  ArrowPathIcon,
  TrashIcon,
} from '@heroicons/react/24/outline'

import { useCapsule } from '@/hooks/useCapsule'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { clsx } from 'clsx'

interface CapsuleCardProps {
  cid: string
  className?: string
  showActions?: boolean
}

export function CapsuleCard({ cid, className, showActions = true }: CapsuleCardProps) {
  const { capsule, isLoading, error, evolveCapsule, isEvolving } = useCapsule(cid)

  if (isLoading) {
    return (
      <div className={clsx('card p-6', className)}>
        <div className="flex items-center justify-center h-32">
          <LoadingSpinner />
        </div>
      </div>
    )
  }

  if (error || !capsule) {
    return (
      <div className={clsx('card p-6 border-red-200 dark:border-red-800', className)}>
        <div className="text-center">
          <p className="text-red-600 dark:text-red-400">Failed to load capsule</p>
        </div>
      </div>
    )
  }

  const handleEvolve = async () => {
    try {
      await evolveCapsule({
        parent_cid: cid,
        evolution_instructions: 'Improve and enhance the game',
        context: {
          evolution_type: 'improvement',
        },
      })
    } catch (error) {
      console.error('Evolution failed:', error)
    }
  }

  const balanceColor = capsule.manifest.balance > 0.7 ? 'text-green-600' :
                       capsule.manifest.balance > 0.4 ? 'text-yellow-600' :
                       'text-red-600'

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4 }}
      className={clsx('card card-hover p-6', className)}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-1">
            {capsule.manifest.title}
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Created {formatDistanceToNow(new Date(capsule.metadata.created_at), { addSuffix: true })}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {capsule.manifest.genre && (
            <span className="px-2 py-1 text-xs font-medium bg-primary-100 text-primary-700 dark:bg-primary-900/30 dark:text-primary-300 rounded-full">
              {capsule.manifest.genre}
            </span>
          )}
          <div className="w-8 h-8 bg-gradient-to-r from-primary-500 to-accent-500 rounded-full flex items-center justify-center">
            <span className="text-white text-xs font-bold">
              {capsule.manifest.title.charAt(0).toUpperCase()}
            </span>
          </div>
        </div>
      </div>

      {/* Description */}
      <p className="text-gray-600 dark:text-gray-300 text-sm mb-4 line-clamp-3">
        {capsule.manifest.story}
      </p>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="flex items-center gap-2">
          <ChartBarIcon className="h-4 w-4 text-gray-500" />
          <div>
            <p className="text-xs text-gray-500">Balance</p>
            <p className={clsx('text-sm font-medium', balanceColor)}>
              {(capsule.manifest.balance * 100).toFixed(0)}%
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <CogIcon className="h-4 w-4 text-gray-500" />
          <div>
            <p className="text-xs text-gray-500">Components</p>
            <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
              {capsule.manifest.llm_outputs.length}
            </p>
          </div>
        </div>
      </div>

      {/* Component Indicators */}
      <div className="flex items-center gap-2 mb-4">
        {capsule.manifest.llm_outputs.map((output, index) => {
          const iconMap = {
            narrative: BookOpenIcon,
            mechanics: CogIcon,
            assets: SparklesIcon,
            balance: ChartBarIcon,
          }

          const Icon = iconMap[output.llm_role as keyof typeof iconMap]
          const confidence = output.confidence || 0.5

          return (
            <div
              key={index}
              className="flex items-center gap-1 px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded-full"
              title={`${output.llm_name}: ${output.llm_role}`}
            >
              <Icon className="h-3 w-3 text-gray-600 dark:text-gray-400" />
              <div
                className="w-2 h-2 rounded-full"
                style={{
                  backgroundColor: confidence > 0.8 ? '#10b981' :
                                     confidence > 0.6 ? '#f59e0b' : '#ef4444'
                }}
              />
            </div>
          )
        })}
      </div>

      {/* Actions */}
      {showActions && (
        <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
          <Link
            to={`/games/${cid}`}
            className="btn-primary flex items-center gap-2 text-sm px-4 py-2"
          >
            <BookOpenIcon className="h-4 w-4" />
            View Game
          </Link>

          <div className="flex items-center gap-2">
            <button
              onClick={handleEvolve}
              disabled={isEvolving}
              className="btn-outline flex items-center gap-2 text-sm px-3 py-2 disabled:opacity-50"
              title="Evolve this game"
            >
              {isEvolving ? (
                <LoadingSpinner size="sm" />
              ) : (
                <ArrowPathIcon className="h-4 w-4" />
              )}
              Evolve
            </button>
          </div>
        </div>
      )}

      {/* CID */}
      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <p className="text-xs text-gray-500 dark:text-gray-400 font-mono">
          CID: {capsule.metadata.game_cid}
        </p>
      </div>
    </motion.div>
  )
}