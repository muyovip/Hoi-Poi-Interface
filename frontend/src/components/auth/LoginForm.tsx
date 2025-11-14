/**
 * Login Form Component
 *
 * Comprehensive login form with validation, error handling, and social login options.
 * Integrates with the authentication system for user login.
 */

import { useState } from 'react'
import { motion } from 'framer-motion'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { Link, useNavigate } from 'react-router-dom'
import { EyeIcon, EyeSlashIcon, EnvelopeIcon, LockClosedIcon } from '@heroicons/react/24/outline'
import { toast } from 'react-hot-toast'

import { useAuthReal } from '@/hooks/useAuthReal'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'

const loginSchema = z.object({
  email: z.string().email('Please enter a valid email address'),
  password: z.string().min(6, 'Password must be at least 6 characters'),
  rememberMe: z.boolean().default(false),
})

type LoginFormData = z.infer<typeof loginSchema>

interface LoginFormProps {
  onLoginSuccess?: () => void
  redirectTo?: string
  className?: string
}

export function LoginForm({ onLoginSuccess, redirectTo, className }: LoginFormProps) {
  const navigate = useNavigate()
  const { login, isLoading } = useAuthReal()
  const [showPassword, setShowPassword] = useState(false)

  const {
    register,
    handleSubmit,
    formState: { errors, isValid },
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      email: '',
      password: '',
      rememberMe: false,
    },
  })

  const onSubmit = async (data: LoginFormData) => {
    try {
      await login(data)
      toast.success('Login successful!')
      onLoginSuccess?.()

      if (redirectTo) {
        navigate(redirectTo)
      } else {
        navigate('/games')
      }
    } catch (error) {
      console.error('Login error:', error)
      toast.error(error instanceof Error ? error.message : 'Login failed')
    }
  }

  const handleSocialLogin = async (provider: 'google' | 'github') => {
    try {
      // In a real implementation, this would redirect to OAuth provider
      window.location.href = `/api/v1/auth/${provider}`
    } catch (error) {
      console.error(`${provider} login error:`, error)
      toast.error(`${provider} login failed`)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={className}
    >
      <div className="card p-8 max-w-md w-full">
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Welcome Back
          </h2>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
            Sign in to your CapsuleOS account
          </p>
        </div>

        {/* Social Login Options */}
        <div className="space-y-3 mb-6">
          <button
            type="button"
            onClick={() => handleSocialLogin('google')}
            className="w-full flex items-center justify-center gap-3 px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
              <path fillRule="evenodd" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25-.07a8.72 8.72 0 01-2.92-.6 7.26 7.26 0 013.14 2.1c.2.72.37 1.44.5 2.25.06a9.86 9.86 0 002.63.77c.72.37 1.5.58 2.29.65a8.9 8.9 0 001.83-3.72 8.8 8.8 0 00-4.17-3.64 9.52 9.52 0 00-6.33-2.35c-.5.29-.86-.5-1.8.01-2.58a8.95 8.95 0 00-4.3-1.84 8.86 8.86 0 00-4.3 2.34 9.53 9.53 0 006.36 2.35c.28.57.38 1.2.41 1.97.02-.8.65-1.5 1.12-2.32.12a8.85 8.85 0 00-2.27-.73 9.12 9.12 0 002.63.76c.8.47 1.58.69 2.44.69 1.4 0 2.72-.27 3.78-.83a8.88 8.88 0 003.75-2.36A8.9 8.9 0 0022.56 12.25z" clipRule="evenodd" />
            </svg>
            <span className="text-gray-700 dark:text-gray-300 font-medium">
              Continue with Google
            </span>
          </button>

          <button
            type="button"
            onClick={() => handleSocialLogin('github')}
            className="w-full flex items-center justify-center gap-3 px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
              <path fillRule="evenodd" d="M12 2C6.477 2 2 6.477 2 12s4.477 10 10 10 10-4.477 10-10S17.523 2 12 2zm3.53 7.03l-3.97-3.97L7.03 14.03 11 17l3.53-3.97L21 17l-5.47-5.97z" clipRule="evenodd" />
            </svg>
            <span className="text-gray-700 dark:text-gray-300 font-medium">
              Continue with GitHub
            </span>
          </button>
        </div>

        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-gray-300 dark:border-gray-600" />
          </div>
          <div className="relative flex justify-center text-sm">
            <span className="px-2 bg-white dark:bg-gray-900 text-gray-500">
              Or continue with email
            </span>
          </div>
        </div>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Email address
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <EnvelopeIcon className="h-5 w-5 text-gray-400" />
              </div>
              <input
                {...register('email')}
                type="email"
                id="email"
                className="input-field pl-10"
                placeholder="you@example.com"
                disabled={isLoading}
              />
            </div>
            {errors.email && (
              <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.email.message}</p>
            )}
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Password
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <LockClosedIcon className="h-5 w-5 text-gray-400" />
              </div>
              <input
                {...register('password')}
                type={showPassword ? 'text' : 'password'}
                id="password"
                className="input-field pl-10 pr-10"
                placeholder="••••••••••"
                disabled={isLoading}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute inset-y-0 right-0 pr-3 flex items-center"
              >
                {showPassword ? (
                  <EyeSlashIcon className="h-5 w-5 text-gray-400" />
                ) : (
                  <EyeIcon className="h-5 w-5 text-gray-400" />
                )}
              </button>
            </div>
            {errors.password && (
              <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.password.message}</p>
            )}
          </div>

          <div className="flex items-center">
            <input
              {...register('rememberMe')}
              type="checkbox"
              id="rememberMe"
              className="h-4 w-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
              disabled={isLoading}
            />
            <label htmlFor="rememberMe" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
              Remember me
            </label>
          </div>

          <div>
            <button
              type="submit"
              disabled={!isValid || isLoading}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <LoadingSpinner size="sm" />
                  Signing in...
                </>
              ) : (
                'Sign in'
              )}
            </button>
          </div>

          <div className="text-center">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Don't have an account?{' '}
              <Link
                to="/register"
                className="font-medium text-primary-600 hover:text-primary-500"
              >
                Sign up for free
              </Link>
            </p>
          </div>

          <div className="text-center">
            <Link
              to="/forgot-password"
              className="text-sm text-primary-600 hover:text-primary-500"
            >
              Forgot your password?
            </Link>
          </div>
        </form>
      </div>
    </motion.div>
  )
}