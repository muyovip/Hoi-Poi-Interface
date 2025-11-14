import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import {
  SparklesIcon,
  BookOpenIcon,
  PlayIcon,
  RocketLaunchIcon,
  CpuChipIcon,
  BeakerIcon
} from '@heroicons/react/24/outline'

export function HomePage() {
  return (
    <div className="space-y-16">
      {/* Hero Section */}
      <section className="text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="max-w-4xl mx-auto"
        >
          <h1 className="text-4xl sm:text-6xl font-bold gradient-text mb-6">
            Multi-LLM Game Generation
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-2xl mx-auto">
            Harness the power of 4 specialized AI models to create unique, balanced, and engaging games through CapsuleOS immutable graph patterns.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              to="/generate"
              className="btn-primary flex items-center justify-center gap-2 px-8 py-3 text-lg"
            >
              <SparklesIcon className="h-5 w-5" />
              Generate a Game
            </Link>
            <Link
              to="/planet"
              className="btn-outline flex items-center justify-center gap-2 px-8 py-3 text-lg"
            >
              <PlayIcon className="h-5 w-5" />
              Explore Planet
            </Link>
          </div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-4">
            Powered by Advanced AI
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Our multi-LLM orchestration system combines specialized models for comprehensive game creation
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={feature.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 * index }}
              className="card card-hover p-6 text-center"
            >
              <div className="flex justify-center mb-4">
                <div className="p-3 bg-primary-100 dark:bg-primary-900/30 rounded-full">
                  <feature.icon className="h-8 w-8 text-primary-600 dark:text-primary-400" />
                </div>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
                {feature.name}
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-gradient-to-r from-primary-50 to-accent-50 dark:from-primary-900/20 dark:to-accent-900/20 rounded-2xl p-8 lg:p-12 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <RocketLaunchIcon className="h-16 w-16 text-primary-600 dark:text-primary-400 mx-auto mb-6" />
          <h2 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-4">
            Ready to Create Something Amazing?
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-300 mb-8 max-w-2xl mx-auto">
            Start generating unique games with our AI-powered system. No coding required – just your imagination.
          </p>
          <Link
            to="/generate"
            className="btn-primary inline-flex items-center gap-2 px-8 py-3 text-lg"
          >
            <SparklesIcon className="h-5 w-5" />
            Start Creating
          </Link>
        </motion.div>
      </section>
    </div>
  )
}

const features = [
  {
    name: 'Narrative Generation',
    description: 'Phi-3 crafts compelling stories, characters, and lore for immersive gaming experiences.',
    icon: BeakerIcon,
  },
  {
    name: 'Game Mechanics',
    description: 'Gemma-2B designs balanced rules, systems, and gameplay mechanics that work seamlessly.',
    icon: CpuChipIcon,
  },
  {
    name: 'Asset Creation',
    description: 'TinyLlama generates detailed asset specifications for sprites, sounds, and visual elements.',
    icon: SparklesIcon,
  },
  {
    name: 'Balance Testing',
    description: 'Qwen-0.5B analyzes and optimizes game balance for fair and engaging gameplay.',
    icon: BookOpenIcon,
  },
  {
    name: 'Immutable Storage',
    description: 'CapsuleOS ensures your games are stored as immutable capsules with verifiable lineage.',
    icon: RocketLaunchIcon,
  },
  {
    name: 'GΛLYPH Expressions',
    description: 'Functional programming expressions enable deterministic and reproducible game generation.',
    icon: CpuChipIcon,
  },
]