import { useState, useEffect, useCallback } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { authManager } from '@/lib/auth'
import type { User, LoginCredentials, RegisterData, UpdateProfileData, UpdatePasswordData } from '@/lib/auth'

export function useAuthReal() {
  const queryClient = useQueryClient()
  const [authState, setAuthState] = useState(authManager.getState())

  // Subscribe to auth manager changes
  useEffect(() => {
    const unsubscribe = authManager.subscribe(() => {
      setAuthState(authManager.getState())
    })

    return unsubscribe
  }, [])

  // Sync with React Query for consistency
  useEffect(() => {
    queryClient.setQueryData(['auth', 'user'], authState.user)
    queryClient.setQueryData(['auth', 'isLoading'], authState.isLoading)
    queryClient.setQueryData(['auth', 'isAuthenticated'], authState.isAuthenticated)
    queryClient.setQueryData(['auth', 'error'], authState.error)
  }, [authState, queryClient])

  const login = useCallback(async (credentials: LoginCredentials) => {
    try {
      await authManager.login(credentials)
    } catch (error) {
      throw error
    }
  }, [])

  const register = useCallback(async (data: RegisterData) => {
    try {
      await authManager.register(data)
    } catch (error) {
      throw error
    }
  }, [])

  const logout = useCallback(async () => {
    try {
      await authManager.logout()
      queryClient.clear()
    } catch (error) {
      console.error('Logout error:', error)
    }
  }, [queryClient])

  const updateProfile = useCallback(async (data: UpdateProfileData) => {
    try {
      const updatedUser = await authManager.updateProfile(data)
      queryClient.setQueryData(['auth', 'user'], updatedUser)
      return updatedUser
    } catch (error) {
      throw error
    }
  }, [queryClient])

  const updatePassword = useCallback(async (data: UpdatePasswordData) => {
    try {
      await authManager.updatePassword(data)
    } catch (error) {
      throw error
    }
  }, [])

  const refreshToken = useCallback(async () => {
    try {
      return await authManager.refreshToken()
    } catch (error) {
      console.error('Token refresh error:', error)
      return false
    }
  }, [])

  return {
    user: authState.user,
    isLoading: authState.isLoading,
    isAuthenticated: authState.isAuthenticated,
    error: authState.error,
    login,
    register,
    logout,
    updateProfile,
    updatePassword,
    refreshToken,

    // Utility methods
    hasRole: authManager.hasRole.bind(authManager),
    hasFeature: authManager.hasFeature.bind(authManager),
    canCreateGame: authManager.canCreateGame.bind(authManager),
    getSubscriptionPlan: authManager.getSubscriptionPlan.bind(authManager),
  }
}