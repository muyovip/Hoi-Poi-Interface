import { useState, useEffect, useCallback } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'

interface User {
  id: string
  email: string
  name?: string
  avatar?: string
  createdAt: string
}

interface AuthState {
  user: User | null
  isLoading: boolean
  isAuthenticated: boolean
}

// Mock auth service - replace with real implementation
const authService = {
  async getCurrentUser(): Promise<User | null> {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500))

    const token = localStorage.getItem('auth_token')
    if (!token) return null

    // Mock user data
    return {
      id: 'user-123',
      email: 'user@example.com',
      name: 'Demo User',
      createdAt: new Date().toISOString(),
    }
  },

  async login(email: string, password: string): Promise<{ user: User; token: string }> {
    // Mock login - replace with real API call
    await new Promise(resolve => setTimeout(resolve, 1000))

    if (email === 'demo@example.com' && password === 'demo') {
      const user = {
        id: 'user-123',
        email: 'demo@example.com',
        name: 'Demo User',
        createdAt: new Date().toISOString(),
      }

      const token = 'mock-jwt-token'
      localStorage.setItem('auth_token', token)

      return { user, token }
    }

    throw new Error('Invalid credentials')
  },

  async logout(): Promise<void> {
    localStorage.removeItem('auth_token')
    await new Promise(resolve => setTimeout(resolve, 300))
  },

  onAuthChange(callback: (user: User | null) => void) {
    // In a real app, this would listen to auth state changes
    return () => {} // Cleanup function
  },
}

export function useAuth() {
  const queryClient = useQueryClient()

  const {
    data: user,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['auth', 'user'],
    queryFn: authService.getCurrentUser,
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: false,
  })

  const login = useCallback(async (email: string, password: string) => {
    try {
      const result = await authService.login(email, password)
      queryClient.setQueryData(['auth', 'user'], result.user)
      return result
    } catch (error) {
      throw error
    }
  }, [queryClient])

  const logout = useCallback(async () => {
    try {
      await authService.logout()
      queryClient.setQueryData(['auth', 'user'], null)
      queryClient.invalidateQueries({ queryKey: ['auth'] })
    } catch (error) {
      console.error('Logout error:', error)
    }
  }, [queryClient])

  const refetchUser = useCallback(() => {
    return refetch()
  }, [refetch])

  return {
    user: user || null,
    isLoading,
    isAuthenticated: !!user,
    error,
    login,
    logout,
    refetchUser: refetchUser,
  }
}